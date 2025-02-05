import os

class Config:
    """Configuration settings for the app."""
    
    # Database settings
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "product_db"
    
    # FAISS settings
    VECTOR_DIM = 512  # Embedding size

    # Model settings
    DEVICE = "cuda" if os.getenv("USE_CUDA", "True") == "True" else "cpu"
