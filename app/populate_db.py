from app.model import CLIPModel
from app.vector_search import VectorSearch

model = CLIPModel()
vector_search = VectorSearch()

sample_products = [
    {"name": "Nike Air Max", "category": "Shoes", "price": 120, "image": "images/nike.jpg"},
    {"name": "Adidas Ultraboost", "category": "Shoes", "price": 150, "image": "images/adidas.jpg"}
]

for product in sample_products:
    embedding = model.get_image_embedding(product["image"])
    vector_search.add_product(embedding, product)

print("Sample product data populated in MongoDB and FAISS!")
