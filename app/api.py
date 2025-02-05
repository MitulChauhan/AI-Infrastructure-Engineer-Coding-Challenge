from fastapi import FastAPI, UploadFile
from app.model import CLIPModel
from app.vector_search import VectorSearch

app = FastAPI()
model = CLIPModel()
vector_search = VectorSearch()

@app.post("/upload")
async def upload_product(name: str, category: str, price: float, image: UploadFile):
    """Upload a product image and store its embedding."""
    image_path = f"images/{image.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    embedding = model.get_image_embedding(image_path)
    metadata = {"name": name, "category": category, "price": price, "image": image_path}
    vector_search.add_product(embedding, metadata)
    
    return {"message": "Product added successfully"}

@app.post("/match")
async def match_product(image: UploadFile = None, query_text: str = None):
    """Find similar products based on image or text query."""
    if image:
        image_path = f"images/{image.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        query_embedding = model.get_image_embedding(image_path)
    elif query_text:
        query_embedding = model.get_text_embedding(query_text)
    else:
        return {"error": "Provide either an image or text query"}

    results = vector_search.find_similar(query_embedding)
    return {"results": results}
