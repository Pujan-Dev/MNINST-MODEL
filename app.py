from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import io

# 1. Define model
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

# 2. Load model and scaler
model = DigitClassifier()
model.load_state_dict(torch.load("digits_model.pth"))
model.eval()

scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

app = FastAPI()

# 3. Image preprocessing function
def process_image(image: Image.Image) -> torch.Tensor:
    image = ImageOps.grayscale(image)           # Convert to grayscale
    image = image.resize((8, 8))                # Resize to 8x8
    image_np = np.array(image)

    # Normalize pixel range from [0-255] to [0-16] to match sklearn digits
    image_np = (image_np / 255.0) * 16.0
    image_np = image_np.reshape(1, -1)

    # Apply same standard scaling as during training
    image_np = (image_np - scaler_mean) / scaler_scale

    return torch.tensor(image_np, dtype=torch.float32)

# 4. Endpoint
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed = process_image(image)

    with torch.no_grad():
        outputs = model(processed)
        _, predicted = torch.max(outputs, 1)
    
    return {"predicted_digit": predicted.item()}
