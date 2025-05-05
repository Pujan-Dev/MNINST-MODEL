from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import io
import os

from huggingface_hub import hf_hub_download

# 1. Download files from Hugging Face repo if not already available
REPO_ID = "Pujan-Dev/MNIST"

model_path = hf_hub_download(repo_id=REPO_ID, filename="digits_model.pth")
mean_path = hf_hub_download(repo_id=REPO_ID, filename="scaler_mean.npy")
scale_path = hf_hub_download(repo_id=REPO_ID, filename="scaler_scale.npy")

# 2. Define model
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

# 3. Load model and scaler
model = DigitClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

scaler_mean = np.load(mean_path)
scaler_scale = np.load(scale_path)

app = FastAPI()

# 4. Image preprocessing function
def process_image(image: Image.Image) -> torch.Tensor:
    image = ImageOps.grayscale(image)
    image = image.resize((8, 8))
    image_np = np.array(image)
    image_np = (image_np / 255.0) * 16.0
    image_np = image_np.reshape(1, -1)
    image_np = (image_np - scaler_mean) / scaler_scale
    return torch.tensor(image_np, dtype=torch.float32)

# 5. Prediction endpoint
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed = process_image(image)

    with torch.no_grad():
        outputs = model(processed)
        _, predicted = torch.max(outputs, 1)
    
    return {"predicted_digit": predicted.item()}

@app.get("/")
def index():
    return {"/docs"}

