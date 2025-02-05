import torch
import clip
from PIL import Image

class CLIPModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_image_embedding(self, image_path):
        """Extracts embeddings from an image using CLIP."""
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
        return image_embedding.cpu().numpy()

    def get_text_embedding(self, text):
        """Extracts embeddings from text using CLIP."""
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)
        return text_embedding.cpu().numpy()
