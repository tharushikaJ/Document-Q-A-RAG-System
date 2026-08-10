"""
Central configuration for the RAG system.
Edit these values to tune behavior without touching the core logic.
"""
import os
from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "example_docs"     # where source documents live
INDEX_DIR = BASE_DIR / "index"           # where the FAISS index + metadata are persisted
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

# ---- Embedding model ----
# Runs locally, no API key needed. 384-dim, fast, good quality for a first RAG system.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Chunking ----
CHUNK_SIZE = 500        # target chunk size, in tokens
CHUNK_OVERLAP = 75      # overlap between consecutive chunks, in tokens

# ---- Retrieval ----
TOP_K = 5                # how many chunks to retrieve per query
MIN_SCORE_THRESHOLD = 0.0  # cosine similarity floor (0 disables filtering)

# ---- Generation (Groq API) ----
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # set this in your shell env