import os
import uuid
import hashlib
import torch
import threading
import logging
from typing import Generator, List, Optional, Dict
import gradio as gr
import json
import re

from transformers import TextIteratorStreamer
from langchain.docstore.document import Document
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import DDGS

# Local imports
import config
import utils

# Use the same logger
logger = config.logger

# For default URL usage
DEFAULT_URLS = config.DEFAULT_URLS
CHROMA_PERSIST_DIR = config.CHROMA_PERSIST_DIR


# ============================================
# TemplateAgent
# ============================================
class TemplateAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _postprocess_output(self, raw_output: str) -> str:
        if "</think>" in raw_output:
            return raw_output.split("</think>")[-1].strip()
        return raw_output

    def generate_template(self, question: str) -> str:
        prompt = (
            "You specialize in designing chain-of-thought templates. Provide a structured approach to solve user input "
            "(e.g., 'Step 1: ...') without revealing any actual reasoning or final answer. "
            "This template must be adaptable to any user query and remain purely an outline. "
            f"\n\n{question}\n\n"
            "Return only approach and steps for the reasoning steps and context, with no real details or final solution.\n\n"
            "Template:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return self._postprocess_output(raw_output)

    def clean_content(self, content: str) -> str:
        if not content.strip():
            return ""
        prompt = (
            "Clean and refine the following context to fix spacing/phrasing issues, ensuring the meaning is preserved:\n\n"
            f"{content}\n\nRefined Context:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=4096)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return self._postprocess_output(raw_output)


# ============================================
# CoTAgent
# ============================================
class CoTAgent:
    def __init__(
        self,
        memory: Optional[ConversationBufferMemory] = None,
        model=None,
        tokenizer=None,
        embeddings=None,
        vectorstore=None,
        retriever=None
    ):
        self.logger = logger
        self.model = model if model else config.GLOBAL_MODEL
        self.tokenizer = tokenizer if tokenizer else config.GLOBAL_TOKENIZER
        self.embeddings = embeddings if embeddings else config.GLOBAL_EMBEDDINGS
        self.vectorstore = vectorstore if vectorstore else config.GLOBAL_VECTORSTORE
        self.retriever = retriever if retriever else config.GLOBAL_RETRIEVER

        self.retriever_cache: Dict[str, List] = {}
        self.system_message = (
            "You are an AI assistant designed to solve problems and answer questions with clear, step-by-step reasoning. "
            "Your goal is to provide accurate, comprehensive, and well-justified responses in English, leveraging retrieved context when applicable.\n\n"
            "Provide a detailed and accurate answer, but avoid repeating information unless it adds new insight. Summarize key points concisely in your final response.\n"
            "During your reasoning, you can take the following actions:\n"
            "- <action>retrieve[query]</action> to retrieve additional info.\n"
            "- <action>generate_template</action> to generate a reasoning template.\n"
            "- <refine>...</refine> to refine a piece of text.\n"
            "When ready to finalize: <final_answer>some answer</final_answer>."
        )

        self.memory = memory if memory else ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if not self.memory.buffer_as_str:
            self.memory.chat_memory.messages.insert(0, SystemMessage(content=self.system_message))

        self.current_urls: List[str] = []
        self.update_urls(DEFAULT_URLS)

        self.template_agent = TemplateAgent(self.model, self.tokenizer)

    def _process_documents(self, docs: List[Document]):
        new_docs = []
        for doc in docs:
            # Use adaptive splitting from utils
            chunks = utils.adaptive_sentence_based_split(doc.page_content, max_tokens=512)
            for chunk in chunks:
                clean_content = chunk.strip()
                content_hash = hashlib.sha256(clean_content.encode()).hexdigest()

                new_doc = Document(
                    page_content=clean_content,
                    metadata={**doc.metadata, "content_hash": content_hash}
                )
                if "source" not in new_doc.metadata:
                    new_doc.metadata["source"] = "unknown"
                new_docs.append(new_doc)

        existing_hashes = set()
        if self.vectorstore._collection.count() > 0:
            existing_data = self.vectorstore._collection.get(include=["metadatas"])
            existing_hashes = {m["content_hash"] for m in existing_data["metadatas"]}

        final_docs = [d for d in new_docs if d.metadata["content_hash"] not in existing_hashes]
        if final_docs:
            self.logger.info(f"🆕 Adding {len(final_docs)} new documents")
            self.vectorstore.add_documents(final_docs)

        doc_count = self.vectorstore._collection.count()
        k = min(4, doc_count) if doc_count > 0 else 1
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def update_urls(self, new_urls: List[str]):
        if set(new_urls) != set(self.current_urls):
            self.logger.info("🔄 Updating document sources")
            self.current_urls = new_urls
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(
                web_paths=new_urls,
                requests_kwargs={"headers": {"User-Agent": f"rag-agent/{uuid.uuid4()}"}}
            )
            docs = loader.load()
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = "url"
            self._process_documents(docs)
        else:
            self.logger.info("✅ URLs unchanged")

    def ingest_pdf_documents(self, file_paths: List[str]):
        from langchain_community.document_loaders import PyPDFLoader
        all_docs = []
        for path in file_paths:
            self.logger.info(f"Ingesting PDF: {path}")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                all_docs.extend(docs)
            except Exception as e:
                self.logger.error(f"Error processing PDF {path}: {e}")
        if all_docs:
            self._process_documents(all_docs)
            return f"Uploaded and processed {len(all_docs)} documents from PDFs."
        return "No valid PDF documents were processed."

    def parse_action(self, text: str):
        # Regex for actions
        template_pattern = r"<action>generate_template</action>"
        refine_pattern = r"<refine>(.*?)</refine>"
        retrieve_pattern = r"<action>retrieve\[(.*?)\]</action>"
        final_answer_pattern = r"<final_answer>(.*?)</final_answer>"

        if re.search(template_pattern, text):
            return "generate_template", None
        elif re.search(refine_pattern, text):
            m = re.search(refine_pattern, text)
            return "refine_context", m.group(1).strip() if m else None
        elif re.search(retrieve_pattern, text):
            m = re.search(retrieve_pattern, text)
            return "retrieve", m.group(1).strip() if m else None
        elif re.search(final_answer_pattern, text):
            return "final_answer", None

        # Check for JSON-like
        try:
            blocks = re.findall(r'\{.*?"command":.*?\}', text, re.DOTALL)
            if blocks:
                last_block = blocks[-1]
                data = json.loads(last_block)
                cmd = data.get("command", {}).get("name")
                if cmd in ["generate_template", "refine_context", "retrieve", "final_answer"]:
                    return cmd, None
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON-like command in text.")

        return None, None

    def search_query(self, query: str, max_results: int):
        results = DDGS().text(query, max_results=max_results)
        return [r["href"] for r in results]

    def _format_response(self, text: str) -> str:
        lines = text.splitlines()
        formatted_lines = []
        in_list = False
        for line in lines:
            line = line.strip()
            if line.startswith(tuple(f"{i}." for i in range(1, 10))):
                step_num = line.split(".")[0]
                step_text = line.split(".", 1)[1].strip()
                formatted_lines.append(f"**Step {step_num}** - {step_text}")
                in_list = True
            elif in_list and line:
                formatted_lines.append(f"- {line}")
            else:
                formatted_lines.append(line)
                in_list = False
        return "\n".join(formatted_lines)

    def generate_with_cot(self, question: str, urls: Optional[List[str]] = None, max_tokens: int = 1024) -> Generator[str, None, None]:
        self.last_question = question
        if not urls or len(urls) == 0:
            urls = self.search_query(question, 8)

        if urls:
            self.update_urls(urls)

        cache_key = hashlib.sha256(question.encode()).hexdigest()
        if cache_key in self.retriever_cache:
            docs = self.retriever_cache[cache_key]
            self.logger.info("Using cached retriever results.")
        else:
            try:
                docs = self.retriever.invoke(question)
                self.retriever_cache[cache_key] = docs
            except Exception as e:
                self.logger.error(f"Error retrieving documents: {e}")
                docs = []

        raw_context = "\n".join(d.page_content for d in docs) if docs else ""
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"Initial Context:\n{raw_context}\n\nQuestion: {question}"}
        ]
        full_response = ""

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            if len(conversation) > 8:
                conversation = [conversation[0]] + conversation[-6:]

            # Qwen chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.2,
                "streamer": streamer,
            }
            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                full_response += new_text
                yield self._format_response(full_response)

                action_type, action_data = self.parse_action(generated_text)
                if action_type == "generate_template":
                    try:
                        template_str = self.template_agent.generate_template(self.last_question)
                        observation = f"Observation: Here is a suggested template: {template_str}"
                    except Exception as e:
                        observation = "Observation: Could not generate template."
                        self.logger.error(f"Template generation error: {e}")
                    conversation.append({"role": "assistant", "content": generated_text})
                    conversation.append({"role": "user", "content": observation})
                    break
                elif action_type == "refine_context":
                    try:
                        refined = self.template_agent.clean_content(action_data)
                        observation = f"Observation: Refined context: {refined}"
                    except Exception as e:
                        observation = f"Observation: Could not refine context. Error: {e}"
                    conversation.append({"role": "assistant", "content": generated_text})
                    conversation.append({"role": "user", "content": observation})
                    break
                elif action_type == "retrieve":
                    try:
                        docs = self.retriever.invoke(action_data)
                        raw_context = "\n".join([d.page_content for d in docs]) if docs else ""
                        cleaned = self.template_agent.clean_content(raw_context)
                        observation = f"Observation: {cleaned}"
                    except Exception as e:
                        observation = f"Observation: Retrieval error: {e}"
                    conversation.append({"role": "assistant", "content": generated_text})
                    conversation.append({"role": "user", "content": observation})
                    break
                elif action_type == "final_answer":
                    thread.join()
                    try:
                        start_idx = generated_text.index("<final_answer>") + len("<final_answer>")
                        end_idx = generated_text.index("</final_answer>")
                        final_ans = generated_text[start_idx:end_idx].strip()
                    except ValueError:
                        final_ans = generated_text.strip()
                    summary = f"\n\n**Summary of Key Points:**\n\n- {final_ans}\n\n"
                    full_response += summary
                    yield self._format_response(full_response)
                    return

            thread.join()
            conversation.append({"role": "assistant", "content": generated_text})

        yield self._format_response(full_response + "\n\n⚠️ Stopped: Maximum iterations reached.")
        return


# ============================================
# Global Agent Registry
# ============================================
AGENT_REGISTRY = {}

def get_agent(session_id: str):
    """
    Return an existing agent or create a new one for this session.
    """
    if session_id not in AGENT_REGISTRY:
        if config.GLOBAL_MODEL is None:
            config.init_global_components()

        from langchain.memory import ConversationBufferMemory
        from langchain_chroma import Chroma

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=config.GLOBAL_EMBEDDINGS
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        agent = CoTAgent(
            memory=memory,
            model=config.GLOBAL_MODEL,
            tokenizer=config.GLOBAL_TOKENIZER,
            embeddings=config.GLOBAL_EMBEDDINGS,
            vectorstore=vectorstore,
            retriever=retriever
        )
        AGENT_REGISTRY[session_id] = agent
    return AGENT_REGISTRY[session_id]


# ============================================
# Gradio Interface
# ============================================
def chat_interface(message: str, urls_text: str, agent_state: any, max_tokens: int) -> Generator[str, None, None]:
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()] if urls_text.strip() else None
    if isinstance(agent_state, dict):
        agent = agent_state.get("agent")
        if agent is None:
            agent = get_agent(str(uuid.uuid4()))
            agent_state["agent"] = agent
    else:
        agent = get_agent(agent_state if agent_state else str(uuid.uuid4()))

    try:
        yield from agent.generate_with_cot(message, urls, max_tokens)
    except Exception as e:
        logger.exception("Generation error:")
        msg = (
            f"❌ An error occurred: {e}\n"
            "Try again or simplify the question. If it persists, check your environment."
        )
        yield msg

def ingest_urls(urls_text: str) -> str:
    if config.GLOBAL_MODEL is None:
        config.init_global_components()
    global_agent = get_agent(str(uuid.uuid4()))

    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    if not urls:
        return "No valid URLs provided."
    global_agent.update_urls(urls)
    return f"Updated with {len(urls)} URL(s)."

def ingest_pdfs(files: List) -> str:
    if config.GLOBAL_MODEL is None:
        config.init_global_components()
    global_agent = get_agent(str(uuid.uuid4()))

    file_paths = []
    os.makedirs("temp_uploads", exist_ok=True)
    for f in files or []:
        if isinstance(f, str):
            file_paths.append(f)
        else:
            tmp_path = os.path.join("temp_uploads", f.name)
            with open(tmp_path, "wb") as out_file:
                out_file.write(f.read())
            file_paths.append(tmp_path)

    if not file_paths:
        return "No PDF files were uploaded."
    return global_agent.ingest_pdf_documents(file_paths)


# ============================================
# Launch Gradio
# ============================================
if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(title="AI Chat Assistant with Document Retrieval") as interface:
        gr.Markdown("# AI Chat Assistant with Document Retrieval")
        gr.Markdown("An intelligent assistant that can retrieve information from documents you provide.")

        session_id = gr.State(value=str(uuid.uuid4()))

        with gr.Tabs():
            with gr.Tab("Ask a Question"):
                gr.Markdown("#### Ask me anything! Optionally specify URLs to consider.")
                with gr.Row():
                    with gr.Column(scale=1):
                        user_input = gr.Textbox(lines=4, placeholder="Type your question...", label="Your Question")
                        urls_optional = gr.Textbox(
                            label="Optional URLs (one per line)",
                            lines=3,
                            placeholder="https://example.com/page1\nhttps://example.com/page2"
                        )
                        max_tokens_slider = gr.Slider(512, 8192, step=256, value=1024, label="Max Tokens")
                        submit_btn = gr.Button("Get Answer")
                        reset_btn = gr.Button("New Chat")
                    with gr.Column(scale=2):
                        answer_output = gr.Markdown()
                        feedback = gr.Slider(1, 5, step=1, label="Helpfulness? (1=bad,5=best)")

                submit_btn.click(fn=chat_interface, inputs=[user_input, urls_optional, session_id, max_tokens_slider],
                                 outputs=answer_output)
                reset_btn.click(lambda s: (str(uuid.uuid4()), ""), session_id, [session_id, answer_output])

            with gr.Tab("Add Web Pages"):
                gr.Markdown("### Add Web Pages for the Assistant to Use")
                url_input = gr.Textbox(label="URLs", lines=5)
                ingest_button = gr.Button("Add URLs")
                ingest_status = gr.Textbox(label="Status")

                ingest_button.click(fn=ingest_urls, inputs=url_input, outputs=ingest_status)

            with gr.Tab("Upload PDFs"):
                gr.Markdown("### Upload PDFs for the Assistant to Reference")
                pdf_files = gr.File(label="Select PDFs", file_count="multiple", file_types=[".pdf"])
                pdf_btn = gr.Button("Upload PDFs")
                pdf_status = gr.Textbox(label="Upload Status")

                pdf_btn.click(fn=ingest_pdfs, inputs=pdf_files, outputs=pdf_status)

        interface.css = """
        body { font-family: 'Arial', sans-serif; }
        .input-box textarea { font-size: 16px !important; padding: 15px !important; border-radius: 8px; }
        .gr-button { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px 20px; }
        .gr-button:hover { background-color: #45a049; }
        .gr-slider { margin-top: 10px; }
        @media (max-width: 768px) { .gr-row { flex-direction: column; } .gr-column { width: 100% !important; } }
        """

    # Start
    try:
        interface.launch(server_name="0.0.0.0", server_port=8000, show_error=True)
    except OSError as e:
        logger.error(f"Port error: {e}")
        interface.launch(server_name="0.0.0.0", server_port=0, show_error=True)
