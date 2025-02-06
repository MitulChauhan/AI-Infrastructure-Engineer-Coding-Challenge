import torch
import clip
from PIL import Image
import os

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Sample Test Image
test_image_path = "images/sample.jpg"

if os.path.exists(test_image_path):
    image = Image.open(test_image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
    
    print("Image Embedding Generated:", image_embedding.shape)
else:
    print("Test image not found!")
