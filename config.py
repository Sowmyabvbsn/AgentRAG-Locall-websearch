import os
import logging
import spacy
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import ollama

# ================================
# GLOBAL SETTINGS & INIT
# ================================

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load spaCy for text processing
GLOBAL_NLP = spacy.load("en_core_web_sm")

# Ollama Model Configuration
OLLAMA_MODEL_NAME = "llama3.2:1b"  # Much smaller and faster model
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "chroma_db"
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# Globals used by main
GLOBAL_MODEL = None
GLOBAL_EMBEDDINGS = None
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None
GLOBAL_TOKENIZER = None

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        ollama.list()
        logger.info("✅ Ollama connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        logger.error("Please ensure Ollama is running: 'ollama serve'")
        return False

def ensure_models_available():
    """Ensure required models are available in Ollama"""
    try:
        models = ollama.list()
        # Handle different response formats from ollama.list()
        if isinstance(models, dict) and 'models' in models:
            model_names = [model.get('name', model.get('model', '')) for model in models['models']]
        else:
            # Fallback for different API response format
            model_names = [getattr(model, 'name', getattr(model, 'model', '')) for model in models]
        
        # Check for Llama2
        if not any("llama3.2:1b" in name for name in model_names):
            logger.info(f"📥 Pulling {OLLAMA_MODEL_NAME} model...")
            ollama.pull(OLLAMA_MODEL_NAME)
            logger.info(f"✅ {OLLAMA_MODEL_NAME} model ready")
        
        # Check for embedding model
        if not any(OLLAMA_EMBEDDING_MODEL in name for name in model_names):
            logger.info(f"📥 Pulling {OLLAMA_EMBEDDING_MODEL} model...")
            ollama.pull(OLLAMA_EMBEDDING_MODEL)
            logger.info(f"✅ {OLLAMA_EMBEDDING_MODEL} model ready")
            
    except Exception as e:
        logger.warning(f"Could not verify models automatically: {e}")
        logger.info("Attempting to pull models anyway...")
        try:
            ollama.pull(OLLAMA_MODEL_NAME)
            ollama.pull(OLLAMA_EMBEDDING_MODEL)
            logger.info("✅ Models pulled successfully")
        except Exception as pull_error:
            logger.error(f"Failed to pull models: {pull_error}")
            logger.error("Please manually run: ollama pull llama3.2:1b && ollama pull nomic-embed-text")

def init_global_components():
    """
    Initialize Ollama models, embeddings, and Chroma vector store
    """
    global GLOBAL_MODEL, GLOBAL_EMBEDDINGS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER, GLOBAL_TOKENIZER

    # Check Ollama connection
    if not check_ollama_connection():
        raise ConnectionError("Cannot connect to Ollama. Please start Ollama service.")
    
    # Ensure models are available
    ensure_models_available()
    
    # Initialize a simple tokenizer for text processing
    try:
        import tiktoken
        GLOBAL_TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        # Fallback tokenizer
        class SimpleTokenizer:
            def encode(self, text):
                return text.split()
        GLOBAL_TOKENIZER = SimpleTokenizer()
        logger.info("Using simple tokenizer (install tiktoken for better performance)")
    
    # Initialize Ollama LLM
    GLOBAL_MODEL = OllamaLLM(
        model=OLLAMA_MODEL_NAME,
        temperature=0.3,  # Lower temperature for faster, more focused responses
        num_predict=1024,  # Reduced max tokens for faster generation
        num_ctx=4096,  # Context window
        top_k=10,  # Limit token sampling for speed
        top_p=0.9,  # Nucleus sampling for efficiency
        repeat_penalty=1.1,
        stop=["\n\n\n"],  # Stop on multiple newlines to prevent rambling
    )
    logger.info(f"✅ Ollama LLM ({OLLAMA_MODEL_NAME}) initialized")

    # Initialize Ollama Embeddings
    GLOBAL_EMBEDDINGS = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
    )
    logger.info(f"✅ Ollama Embeddings ({OLLAMA_EMBEDDING_MODEL}) initialized")

    # Initialize Chroma vector store
    GLOBAL_VECTORSTORE = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=GLOBAL_EMBEDDINGS
    )
    
    doc_count = GLOBAL_VECTORSTORE._collection.count()
    k = min(6, doc_count) if doc_count > 0 else 1
    GLOBAL_RETRIEVER = GLOBAL_VECTORSTORE.as_retriever(
        search_kwargs={"k": k, "fetch_k": 10}
    )

    logger.info("🚀 Global components initialized successfully")