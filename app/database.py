import faiss
import numpy as np
from pymongo import MongoClient

class Database:
    def __init__(self):
        # MongoDB Setup
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client["product_db"]
        self.collection = self.db["products"]
        self.logs = self.db["logs"]

        # FAISS Vector DB Setup
        self.d = 512  # Embedding dimension
        self.index = faiss.IndexFlatL2(self.d)

    def insert_product(self, product):
        """Insert a product into MongoDB."""
        self.collection.insert_one(product)

    def log_event(self, log_entry):
        """Log events in MongoDB."""
        self.logs.insert_one(log_entry)

    def add_embedding(self, embedding):
        """Add an embedding to FAISS."""
        self.index.add(np.array([embedding]).astype('float32'))

    def search(self, query_embedding, top_k=5):
        """Search for nearest neighbors in FAISS."""
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return indices
