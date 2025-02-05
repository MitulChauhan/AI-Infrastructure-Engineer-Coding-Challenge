from app.database import Database

class VectorSearch:
    def __init__(self):
        self.db = Database()

    def add_product(self, embedding, metadata):
        """Add product embedding and metadata."""
        self.db.add_embedding(embedding)
        self.db.insert_product(metadata)

    def find_similar(self, query_embedding, top_k=5):
        """Find similar products in FAISS."""
        indices = self.db.search(query_embedding, top_k)
        results = [self.db.collection.find_one({"_id": i}) for i in indices[0]]
        return results
