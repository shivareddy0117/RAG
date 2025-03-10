import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEMANTIC_CHUNKING = True  # Use semantic chunking instead of fixed size

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Default embedding model
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 16

# Vector Database
VECTORDB_TYPE = "FAISS"  # Options: "FAISS", "Chroma", "Pinecone"
SIMILARITY_TOP_K = 5
SIMILARITY_SCORE_THRESHOLD = 0.7

# Model Configuration
LLM_MODEL = "gpt-3.5-turbo"  # Default model
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# Retrieval Configuration
HYBRID_SEARCH = True  # Use hybrid (dense + sparse) search
RERANKING_ENABLED = True  # Enable reranking of retrieved documents

# Hardware Acceleration
USE_GPU = True
MIXED_PRECISION = True  # Use mixed precision (fp16) for inference
CUDA_VISIBLE_DEVICES = "0"  # Specify which GPUs to use

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
VECTORSTORE_PATH = "vectorstore"

# Evaluation
EVAL_METRICS = ["precision", "recall", "f1", "accuracy"]

# Performance Tuning
NUM_THREADS = 4  # Number of threads for parallel processing
BATCH_SIZE = 8  # Batch size for inference

# Logging
LOG_LEVEL = "INFO" 