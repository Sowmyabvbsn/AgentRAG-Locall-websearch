# 🤖 AI Agent Assistant (Local & Free) with Web Search & Document Retrieval (RAG) 🚀

An intelligent assistant you can **run locally for free**, complete with **web search** and **document retrieval**. By default, it uses a quantized Qwen LLM so it fits on most mid-range GPUs. No external API keys required, and your data stays private on your machine.

## 🌟 Key Features

- **🗃️ Auto Download/Load Model**  
  Uses [Hugging Face Transformers](https://github.com/huggingface/transformers) to automatically download or load a local LLM.  
  <sub>_Default:_ [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)_</sub>

- **🔁 Easily Swap LLMs**  
  Change the `MODEL_NAME` in `config.py` to any other model (e.g., [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)) to run on smaller GPUs or CPU.

- **⚙️ Quantized for Mid-range GPUs**  
  By default, the model runs in 8-bit, reducing memory usage so it’s easier to fit on typical RTX GPUs. Switch to a lighter model if your resources are more limited.

- **🤖 Agentic RAG Architecture**  
  The agent can retrieve documents from your vector database and provide responses. It also auto-searches the web for each query when no URLs are specified.

- **🔎 Local Vector Store**  
  Documents (PDFs, web pages) are indexed in a local [Chroma](https://github.com/chroma-core/chroma) database. No data leaves your system.

- **🌐 Web Interface with Gradio**  
  A user-friendly web UI. No external API keys needed.

- **👩‍💻 High Privacy & Local Usage**  
  All logs, models, and vector data remain on your machine. No remote calls.

- **📏 Adjustable Max Tokens**  
  Generate answers from 512 up to 8192 tokens, depending on your preference or system capabilities.

## 🏗️ Project Overview

- **config.py**  
  Manages global settings (model, tokenizer, embeddings, and Chroma vector store).

- **utils.py**  
  Contains text-splitting logic for chunking documents.

- **main.py**  
  Houses the CoTAgent logic, TemplateAgent, and Gradio interface.  

## 📦 Requirements

- Python 3.9+ recommended  
- A GPU with enough VRAM for your chosen model  
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or similar environment manager (optional)

## 🚀 Installation

1. **Clone the Repo**  
   ```bash
   git clone https://github.com/<your-username>/ai-agent-assistant-rag.git
   cd ai-agent-assistant-rag
   ```

2. **(Optional) Create/Activate a Conda Environment**  
   ```bash
   conda create -n ai2 python=3.9
   conda activate ai2
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Model**  
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 💻 Usage

1. **Run the App**  
   ```bash
   python main.py
   ```
   _Default port is 8000. If it’s busy, Gradio picks another._

2. **Access via Browser**  
   Open `http://0.0.0.0:8000` (or the displayed Gradio link) to chat.

3. **Ask Questions**  
   - Type your query.  
   - (Optional) Provide URLs or let it auto-search.  
   - Upload PDFs to expand local knowledge.  
   - Adjust max tokens.  
   - Click **Get Answer**.

## 📂 Folder Structure

```
project/
├── config.py
├── utils.py
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

- **config.py** – global LLM/embeddings init  
- **utils.py** – text-splitting helpers  
- **main.py** – main logic (TemplateAgent, CoTAgent, Gradio UI)

## ❓ Troubleshooting

- **Model Not Found**: Verify `MODEL_PATH` in `config.py` or let it auto-download.  
- **Out of Memory**: Choose a smaller LLM (like [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)).  
- **Chroma Errors**: Remove `chroma_db/` to reset indexing. Confirm the same embedding function is used each time.

## 🔏 License

Released under the [MIT License](LICENSE).

