import os
import logging
import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

# GPU/CPU memory usage
max_memory = {0: "11GB", "cpu": "30GB"}

# Model / Embedding config
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"
MODEL_PATH = os.path.join("../local_model", MODEL_NAME)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_PATH = "../local_embeddings"
CHROMA_PERSIST_DIR = "chroma_db"
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# Globals used by main
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_EMBEDDINGS = None
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None

def init_global_components():
    """
    Loads or downloads the model/tokenizer, sets up embeddings,
    and initializes Chroma's vector store and retriever.
    """
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_EMBEDDINGS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_files_exist = os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json"))
    if model_files_exist:
        try:
            logger.info(f"Loading local model from {MODEL_PATH}")
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                max_memory=max_memory
            )
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            logger.info("✅ Local model loaded")
        except Exception as e:
            logger.warning(f"Local model corrupted or incomplete: {e}. Downloading model.")
            model_files_exist = False

    if not model_files_exist:
        # BitsAndBytes configs for 8-bit
        from transformers import BitsAndBytesConfig
        quant_config_int8 = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.int8
        )
        logger.info(f"Downloading model: {MODEL_NAME}")
        GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quant_config_int8,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        os.makedirs(MODEL_PATH, exist_ok=True)
        GLOBAL_MODEL.save_pretrained(MODEL_PATH)
        GLOBAL_TOKENIZER.save_pretrained(MODEL_PATH)
        logger.info("💾 Model downloaded and cached")

    # Embeddings
    embedding_file = os.path.join(EMBEDDING_MODEL_PATH, "pytorch_model.bin")
    os.makedirs(EMBEDDING_MODEL_PATH, exist_ok=True)
    if not os.path.exists(embedding_file):
        logger.info("⬇️ Downloading embedding model...")
        st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st_model.save(EMBEDDING_MODEL_PATH)
        logger.info("💾 Saved embeddings locally")
    else:
        logger.info("✅ Loading embeddings from local cache")
        st_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"}
    )

    # Chroma store
    GLOBAL_VECTORSTORE = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=GLOBAL_EMBEDDINGS
    )
    doc_count = GLOBAL_VECTORSTORE._collection.count()
    k = min(4, doc_count) if doc_count > 0 else 1
    GLOBAL_RETRIEVER = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": k})

    logger.info("Global components initialized.")
