import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom dataset
class ProductDataset(Dataset):
    def __init__(self, data_dir, preprocess):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare dataset and dataloader
dataset = ProductDataset("images", preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop (Example, CLIP is pretrained, so fine-tuning is optional)
for batch in dataloader:
    batch = batch.to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
    
    print("Batch Processed: ", features.shape)

print("✅ Training complete (if fine-tuning required)")
