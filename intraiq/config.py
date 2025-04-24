"""Configuration settings for the IntraIQ application."""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.environ.get("INTRAIQ_DATA_DIR", ROOT_DIR / "data")
MODELS_DIR = os.environ.get("INTRAIQ_MODELS_DIR", DATA_DIR / "models")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Document processing
DEFAULT_CHUNK_SIZE = 500  # tokens
DEFAULT_CHUNK_OVERLAP = 50  # tokens

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Qdrant settings
QDRANT_LOCAL_PATH = os.environ.get("QDRANT_LOCAL_PATH", str(DATA_DIR / "qdrant"))
QDRANT_URL = os.environ.get("QDRANT_URL", None)
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
DEFAULT_COLLECTION = "intraiq_documents"

# Make sure local Qdrant path exists
if QDRANT_URL is None:
    Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)