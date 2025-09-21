import os
import uuid
import hashlib
import threading
import logging
from typing import Generator, List, Optional, Dict
import gradio as gr
import json
import re
import datetime
from langchain.docstore.document import Document
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import DDGS

# Local imports
import config
import utils
from multimodal_processor import MultiModalProcessor
from citation_manager import CitationManager

# Use the same logger
logger = config.logger

# For default URL usage
DEFAULT_URLS = config.DEFAULT_URLS
CHROMA_PERSIST_DIR = config.CHROMA_PERSIST_DIR


# ============================================
# Enhanced CoTAgent with Ollama and Citations
# ============================================
class EnhancedCoTAgent:
    def __init__(
        self,
        memory: Optional[ConversationBufferMemory] = None,
        model=None,
        embeddings=None,
        vectorstore=None,
        retriever=None
    ):
        self.logger = logger
        self.model = model if model else config.GLOBAL_MODEL
        self.embeddings = embeddings if embeddings else config.GLOBAL_EMBEDDINGS
        self.vectorstore = vectorstore if vectorstore else config.GLOBAL_VECTORSTORE
        self.retriever = retriever if retriever else config.GLOBAL_RETRIEVER

        self.retriever_cache: Dict[str, List] = {}
        self.system_message = (
            "You are an advanced AI assistant powered by Llama2, designed to provide comprehensive, accurate, and well-researched responses. "
            "You have access to a hybrid knowledge base that combines your training data with user-provided documents, images, audio transcriptions, and web content. "
            "Your responses should be:\n"
            "1. Accurate and factual\n"
            "2. Well-structured and easy to understand\n"
            "3. Comprehensive yet concise\n"
            "4. Supported by evidence from your knowledge base\n"
            "5. Include relevant context from retrieved documents when applicable\n\n"
            "When answering questions, consider both your pre-trained knowledge and the retrieved context to provide the most complete and accurate response possible."
        )

        self.memory = memory if memory else ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if not self.memory.buffer_as_str:
            self.memory.chat_memory.messages.insert(0, SystemMessage(content=self.system_message))

        self.current_urls: List[str] = []
        self.update_urls(DEFAULT_URLS)

        # Initialize multimodal processor and citation manager
        self.multimodal_processor = MultiModalProcessor()
        self.citation_manager = CitationManager()

    def _process_documents(self, docs: List[Document]):
        """Process and add documents to vector store with enhanced metadata"""
        from langchain_community.vectorstores.utils import filter_complex_metadata
        new_docs = []
        
        for doc in docs:
            doc_content = doc.page_content

            # Enhanced document type detection
            doc_types = set()
            
            # Detect code
            if "```" in doc_content or any(keyword in doc_content for keyword in ["def ", "import ", "function", "class ", "var ", "let ", "const "]):
                doc_types.add("code")
            
            # Detect math
            if any(math_sym in doc_content for math_sym in ["∫", "∑", "√", "π", "∞", "Δ", "lim", "equation", "formula"]):
                doc_types.add("math")
            
            # Detect academic content
            if any(academic_term in doc_content.lower() for academic_term in ["abstract", "methodology", "conclusion", "references", "citation"]):
                doc_types.add("academic")
            
            # Default to text if no specific type detected
            non_code_text = re.sub(r"```(.*?)```", "", doc_content, flags=re.DOTALL).strip()
            if non_code_text and len(non_code_text.split()) > 10:
                doc_types.add("text")
            
            if not doc_types:
                doc_types.add("text")

            # Extract title or section header
            content_stripped = doc.page_content.strip()
            candidate_title = ""
            if content_stripped:
                first_line = content_stripped.splitlines()[0].strip()
                if first_line.startswith("#"):
                    candidate_title = first_line.lstrip("#").strip()
                elif len(first_line.split()) <= 10:
                    candidate_title = first_line

            # Use adaptive splitting
            chunks = utils.adaptive_sentence_based_split(doc.page_content, max_tokens=512)
            
            for chunk in chunks:
                clean_content = chunk.strip()
                content_hash = hashlib.sha256(clean_content.encode()).hexdigest()
                ingested_at = datetime.datetime.utcnow().isoformat()

                # Enhanced metadata
                new_metadata = {
                    **doc.metadata,
                    "content_hash": content_hash,
                    "ingested_at": ingested_at,
                    "doc_type": list(doc_types),
                    "chunk_length": len(clean_content),
                    "word_count": len(clean_content.split())
                }
                
                if candidate_title:
                    new_metadata["section_title"] = candidate_title
                if "source" not in new_metadata:
                    new_metadata["source"] = "unknown"

                # Filter complex metadata
                filtered_docs = filter_complex_metadata([Document(page_content="", metadata=new_metadata)])
                new_metadata = filtered_docs[0].metadata

                new_doc = Document(
                    page_content=clean_content,
                    metadata=new_metadata
                )
                new_docs.append(new_doc)

        # Check for existing documents
        existing_hashes = set()
        if self.vectorstore._collection.count() > 0:
            existing_data = self.vectorstore._collection.get(include=["metadatas"])
            existing_hashes = {m.get("content_hash", "") for m in existing_data["metadatas"]}

        final_docs = [d for d in new_docs if d.metadata["content_hash"] not in existing_hashes]
        
        if final_docs:
            self.logger.info(f"🆕 Adding {len(final_docs)} new document chunks")
            self.vectorstore.add_documents(final_docs)

        # Update retriever
        doc_count = self.vectorstore._collection.count()
        k = min(6, doc_count) if doc_count > 0 else 1
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k, "fetch_k": 10, "include_metadata": True}
        )

    def update_urls(self, new_urls: List[str]):
        """Update URLs and process web content"""
        if set(new_urls) != set(self.current_urls):
            self.logger.info("🔄 Updating web document sources")
            self.current_urls = new_urls
            
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(
                web_paths=new_urls,
                requests_kwargs={"headers": {"User-Agent": f"enhanced-rag-agent/{uuid.uuid4()}"}}
            )
            
            try:
                docs = loader.load()
                for doc in docs:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = "web"
                    doc.metadata["type"] = "web_content"
                
                self._process_documents(docs)
                self.logger.info(f"✅ Processed {len(docs)} web documents")
                
            except Exception as e:
                self.logger.error(f"Error loading web content: {e}")
        else:
            self.logger.info("✅ URLs unchanged")

    def ingest_multimodal_files(self, file_paths: List[str]) -> str:
        """Process multiple types of files using multimodal processor"""
        all_docs = []
        processed_files = []
        failed_files = []
        processing_summary = []
        
        for path in file_paths:
            self.logger.info(f"Processing file: {path}")
            try:
                docs = self.multimodal_processor.process_file(path)
                if docs:
                    all_docs.extend(docs)
                    processed_files.append(os.path.basename(path))
                    # Get file type for summary
                    file_ext = os.path.splitext(path)[1].lower()
                    file_type = "unknown"
                    if file_ext == '.pdf':
                        file_type = "PDF"
                    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                        file_type = "Image"
                    elif file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                        file_type = "Audio"
                    elif file_ext in ['.txt', '.md', '.rst']:
                        file_type = "Text"
                    
                    processing_summary.append(f"{os.path.basename(path)} ({file_type}): {len(docs)} chunks")
                else:
                    self.logger.warning(f"No content extracted from {path}")
                    failed_files.append(os.path.basename(path))
                    
            except Exception as e:
                self.logger.error(f"Error processing file {path}: {e}")
                failed_files.append(f"{os.path.basename(path)} (Error: {str(e)[:50]}...)")
        
        if all_docs:
            self._process_documents(all_docs)
            
            result_parts = [
                f"Successfully processed {len(processed_files)} out of {len(file_paths)} files",
                f"Total document chunks created: {len(all_docs)}",
                "",
                "📋 **Processing Details:**"
            ]
            result_parts.extend([f"  • {summary}" for summary in processing_summary])
            
            if failed_files:
                result_parts.extend([
                    "",
                    "⚠️ **Failed Files:**"
                ])
                result_parts.extend([f"  • {failed}" for failed in failed_files])
            
            return "\n".join(result_parts)
        
        return f"❌ No valid files were processed out of {len(file_paths)} files. Check file formats and try again."

    def search_query(self, query: str, max_results: int = 8) -> List[str]:
        """Search for relevant URLs using DuckDuckGo"""
        try:
            results = DDGS().text(query, max_results=max_results)
            return [r["href"] for r in results]
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def generate_with_enhanced_context(self, question: str, urls: Optional[List[str]] = None, max_tokens: int = 2048) -> Generator[str, None, None]:
        """Generate response with enhanced context and citations"""
        self.last_question = question
        
        # Search for relevant URLs if none provided
        if not urls or len(urls) == 0:
            urls = self.search_query(question, 8)

        if urls:
            self.update_urls(urls)

        # Retrieve relevant documents
        cache_key = hashlib.sha256(question.encode()).hexdigest()
        if cache_key in self.retriever_cache:
            docs = self.retriever_cache[cache_key]
            self.logger.info("Using cached retriever results.")
        else:
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=6)
                docs = [doc for doc, score in docs_with_scores]
                self.logger.info("Similarity scores: " + ", ".join(f"{score:.3f}" for _, score in docs_with_scores))
                self.retriever_cache[cache_key] = docs
            except Exception as e:
                self.logger.error(f"Error retrieving documents: {e}")
                docs = []

        # Prepare context from retrieved documents
        context_parts = []
        if docs:
            for i, doc in enumerate(docs, 1):
                doc_type = doc.metadata.get('type', 'unknown')
                source = doc.metadata.get('source', 'unknown')
                context_parts.append(f"[Source {i} - {doc_type} from {source}]:\n{doc.page_content}\n")

        raw_context = "\n".join(context_parts) if context_parts else ""
        
        # Create enhanced prompt
        enhanced_prompt = f"""Based on the following context from various sources and your knowledge, provide a comprehensive answer to the question.

Context from Knowledge Base:
{raw_context}

Question: {question}

Please provide a detailed, accurate response that:
1. Combines information from the provided context with your knowledge
2. Is well-structured and easy to understand
3. Addresses all aspects of the question
4. Mentions relevant sources when applicable

Response:"""

        try:
            # Generate response using Ollama
            response_text = ""
            for chunk in self.model.stream(enhanced_prompt):
                response_text += chunk
                yield response_text
            
            # Generate citations and related links
            citations = self.citation_manager.generate_citations_for_response(response_text, question)
            citation_text = self.citation_manager.format_citations_for_display(citations)
            
            # Add citations to response
            final_response = response_text + citation_text
            yield final_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            yield f"❌ Error generating response: {str(e)}"


# ============================================
# Global Agent Registry
# ============================================
AGENT_REGISTRY = {}

def get_agent(session_id: str):
    """Return an existing agent or create a new one for this session"""
    if session_id not in AGENT_REGISTRY:
        try:
            if config.GLOBAL_MODEL is None:
                config.init_global_components()
        except Exception as e:
            logger.error(f"Failed to initialize global components: {e}")
            raise ConnectionError("Cannot initialize Ollama components. Please ensure Ollama is running and models are available.")

        from langchain.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent = EnhancedCoTAgent(
            memory=memory,
            model=config.GLOBAL_MODEL,
            embeddings=config.GLOBAL_EMBEDDINGS,
            vectorstore=config.GLOBAL_VECTORSTORE,
            retriever=config.GLOBAL_RETRIEVER
        )
        AGENT_REGISTRY[session_id] = agent
    
    return AGENT_REGISTRY[session_id]


# ============================================
# Gradio Interface Functions
# ============================================
def chat_interface(message: str, urls_text: str, agent_state: any, max_tokens: int) -> Generator[str, None, None]:
    """Main chat interface"""
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()] if urls_text.strip() else None
    
    if isinstance(agent_state, dict):
        agent = agent_state.get("agent")
        if agent is None:
            agent = get_agent(str(uuid.uuid4()))
            agent_state["agent"] = agent
    else:
        agent = get_agent(agent_state if agent_state else str(uuid.uuid4()))

    try:
        response_generator = agent.generate_with_enhanced_context(message, urls, max_tokens)
        accumulated_response = ""
        
        for chunk in response_generator:
            accumulated_response = chunk
            yield accumulated_response
            
        # If no response was generated, provide a fallback
        if not accumulated_response.strip():
            yield "I apologize, but I wasn't able to generate a response. Please try rephrasing your question or check that the system is properly configured."
            
    except Exception as e:
        logger.exception("Generation error:")
        msg = (
            f"❌ An error occurred: {e}\n"
            "Please check that Ollama is running and the Llama2 model is available.\n"
            "Run: `ollama serve` and `ollama pull llama2`"
        )
        yield msg

def ingest_urls(urls_text: str) -> str:
    """Ingest URLs into the knowledge base"""
    try:
        if config.GLOBAL_MODEL is None:
            config.init_global_components()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return f"❌ System initialization failed: {str(e)}. Please ensure Ollama is running and models are available."
    
    global_agent = get_agent(str(uuid.uuid4()))
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    
    if not urls:
        return "No valid URLs provided."
    
    try:
        global_agent.update_urls(urls)
        return f"Successfully ingested {len(urls)} URL(s) into the knowledge base."
    except Exception as e:
        logger.error(f"Error ingesting URLs: {e}")
        return f"❌ Error processing URLs: {str(e)}"

def ingest_multimodal_files(files: List) -> str:
    """Ingest various file types (PDFs, images, audio, documents)"""
    try:
        if config.GLOBAL_MODEL is None:
            config.init_global_components()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return f"❌ System initialization failed: {str(e)}. Please ensure Ollama is running and models are available."
    
    global_agent = get_agent(str(uuid.uuid4()))
    
    if not files:
        return "❌ No files were uploaded. Please select one or more files to process."
    
    file_paths = []
    file_info = []
    os.makedirs("temp_uploads", exist_ok=True)
    
    try:
        for f in files or []:
            if isinstance(f, str):
                file_paths.append(f)
                file_info.append(os.path.basename(f))
            else:
                # Handle file objects from Gradio
                if hasattr(f, 'name') and f.name:
                    # f is already a file path from Gradio
                    file_paths.append(f.name)
                    file_info.append(os.path.basename(f.name))
                else:
                    # f is a file-like object, need to save it
                    tmp_path = os.path.join("temp_uploads", getattr(f, 'name', f'uploaded_file_{len(file_paths)}'))
                    with open(tmp_path, "wb") as out_file:
                        out_file.write(f.read())
                    file_paths.append(tmp_path)
                    file_info.append(os.path.basename(tmp_path))
                file_paths.append(tmp_path)
                file_info.append(os.path.basename(tmp_path))
    except Exception as e:
        logger.error(f"Error processing uploaded files: {e}")
        return f"❌ Error processing files: {str(e)}"

    logger.info(f"📁 Processing {len(file_paths)} files: {', '.join(file_info)}")
    
    try:
        result = global_agent.ingest_multimodal_files(file_paths)
        return f"✅ {result}\n\n📊 **Files processed:** {len(file_paths)}\n📝 **File list:** {', '.join(file_info)}"
    except Exception as e:
        logger.error(f"Error ingesting files: {e}")
        return f"❌ Error processing files: {str(e)}"


# ============================================
# Launch Enhanced Gradio Interface
# ============================================
if __name__ == "__main__":
    import gradio as gr

    # Custom CSS for better styling
    custom_css = """
    .main-container { max-width: 1200px; margin: 0 auto; }
    .chat-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .input-box textarea { 
        font-size: 16px !important; 
        padding: 15px !important; 
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        transition: border-color 0.3s ease;
    }
    .input-box textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .gr-button { 
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white; 
        border: none; 
        border-radius: 8px; 
        padding: 12px 24px;
        font-weight: 600;
        transition: transform 0.2s ease;
    }
    .gr-button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .status-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    .file-upload {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .file-upload:hover {
        border-color: #764ba2;
        background-color: #f8f9ff;
    }
    @media (max-width: 768px) { 
        .gr-row { flex-direction: column; } 
        .gr-column { width: 100% !important; } 
    }
    """

    with gr.Blocks(title="🤖 Enhanced RAG Assistant with Llama2", css=custom_css, theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #333; margin-bottom: 10px;">🤖 Enhanced RAG Assistant</h1>
            <h2 style="color: #666; font-weight: 300;">Powered by Ollama Llama2 with Multi-Modal Support</h2>
            <p style="color: #888;">Upload PDFs, images, audio files, or provide URLs to build your knowledge base</p>
        </div>
        """)

        session_id = gr.State(value=str(uuid.uuid4()))

        with gr.Tabs():
            with gr.Tab("💬 Chat Assistant", elem_classes=["chat-container"]):
                gr.Markdown("### Ask me anything! I can access your uploaded documents, images, audio, and web content.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        user_input = gr.Textbox(
                            lines=4, 
                            placeholder="Ask me about your documents, or any topic...", 
                            label="Your Question",
                            elem_classes=["input-box"]
                        )
                        urls_optional = gr.Textbox(
                            label="Optional URLs (one per line)",
                            lines=3,
                            placeholder="https://example.com/article1\nhttps://example.com/article2",
                            elem_classes=["input-box"]
                        )
                        max_tokens_slider = gr.Slider(
                            256, 2048, 
                            step=128, 
                            value=1024, 
                            label="Max Response Length"
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("🚀 Get Answer", variant="primary")
                            reset_btn = gr.Button("🔄 New Chat", variant="secondary")
                    
                    with gr.Column(scale=2):
                        answer_output = gr.Markdown(label="Response")
                        
                        with gr.Row():
                            feedback = gr.Slider(
                                1, 5, 
                                step=1, 
                                label="Rate this response (1=Poor, 5=Excellent)",
                                visible=False
                            )

                submit_btn.click(
                    fn=chat_interface, 
                    inputs=[user_input, urls_optional, session_id, max_tokens_slider],
                    outputs=answer_output,
                    show_progress=True
                )
                
                reset_btn.click(
                    lambda s: (str(uuid.uuid4()), ""), 
                    inputs=session_id, 
                    outputs=[session_id, answer_output]
                )

            with gr.Tab("🌐 Add Web Content"):
                gr.Markdown("### 📚 Add Web Pages to Knowledge Base")
                gr.Markdown("Provide URLs to articles, documentation, or any web content you want the assistant to reference.")
                
                url_input = gr.Textbox(
                    label="URLs (one per line)", 
                    lines=8,
                    placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://arxiv.org/abs/2103.00020",
                    elem_classes=["input-box"]
                )
                ingest_button = gr.Button("📥 Add URLs to Knowledge Base", variant="primary")
                ingest_status = gr.Textbox(label="Status", elem_classes=["status-box"])

                ingest_button.click(fn=ingest_urls, inputs=url_input, outputs=ingest_status)

            with gr.Tab("📁 Upload Files"):
                gr.Markdown("### 📎 Upload Multi-Modal Content")
                gr.Markdown("""
                **Supported file types:**
                - 📄 **Documents**: PDF, TXT, MD
                - 🖼️ **Images**: JPG, PNG, BMP, TIFF, WEBP (text extraction + image analysis)
                - 🎵 **Audio**: MP3, WAV, M4A, FLAC, OGG (speech will be transcribed)
                
                **Tips for best results:**
                - Upload multiple files at once for batch processing
                - Mix different file types (PDFs + images + audio) for comprehensive knowledge base
                - Ensure images contain clear, readable text for OCR
                - Audio files should have clear speech for accurate transcription
                """)
                
                multimodal_files = gr.File(
                    label="Select Multiple Files", 
                    file_count="multiple", 
                    file_types=[".pdf", ".txt", ".md", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                    height=200
                )
                
                with gr.Row():
                    clear_files_btn = gr.Button("🗑️ Clear Files", variant="secondary")
                    
                upload_btn = gr.Button("🚀 Process Files", variant="primary")
                upload_status = gr.Textbox(label="Processing Status", elem_classes=["status-box"])

                upload_btn.click(fn=ingest_multimodal_files, inputs=multimodal_files, outputs=upload_status)
                clear_files_btn.click(lambda: None, outputs=multimodal_files)

            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("""
                ### 🔧 System Information
                
                **Model**: Ollama Llama 3.2 1B (Optimized for Speed)  
                **Embeddings**: Nomic Embed Text  
                **Vector Store**: Chroma DB  
                **Capabilities**:
                - 📝 Text processing and understanding
                - 🖼️ Image text extraction (OCR) + visual analysis
                - 🎵 Audio transcription
                - 🌐 Web content ingestion
                - 📚 Citation generation
                - 🔗 Related link recommendations
                - 📁 **Batch file processing** - Upload multiple files simultaneously
                - 🔄 **Mixed media support** - Process PDFs, images, and audio together
                
                **Requirements**:
                - Ollama must be running (`ollama serve`)
                - Llama 3.2 1B model must be available (`ollama pull llama3.2:1b`)
                - Nomic embedding model (`ollama pull nomic-embed-text`)
                
                **File Processing Limits:**
                - Maximum file size: Depends on available system memory
                - Supported formats: PDF, TXT, MD, JPG, PNG, BMP, TIFF, WEBP, MP3, WAV, M4A, FLAC, OGG
                - Batch processing: Upload multiple files simultaneously for efficient processing
                """)
                
                # System status check
                def check_system_status():
                    try:
                        if config.check_ollama_connection():
                            try:
                                # Try to initialize components
                                if config.GLOBAL_MODEL is None:
                                    config.init_global_components()
                                return "✅ System is ready! Ollama is running and models are available."
                            except Exception as e:
                                return f"⚠️ Ollama is running but models may not be available: {str(e)}\nTry running: ollama pull llama3.2:1b && ollama pull nomic-embed-text"
                        else:
                            return "❌ Cannot connect to Ollama. Please ensure Ollama is running with: ollama serve"
                    except Exception as e:
                        return f"❌ System check failed: {str(e)}"
                
                status_btn = gr.Button("🔍 Check System Status")
                system_status = gr.Textbox(label="System Status")
                status_btn.click(fn=check_system_status, outputs=system_status)

    # Launch the interface
    try:
        interface.launch(
            server_name="0.0.0.0", 
            server_port=8000, 
            show_error=True, 
            share=False,  # Set to True if you want a public link
            inbrowser=True
        )
    except OSError as e:
        logger.error(f"Port 8000 is busy: {e}")
        interface.launch(
            server_name="0.0.0.0", 
            server_port=0, 
            show_error=True,
            inbrowser=True
        )