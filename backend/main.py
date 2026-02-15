# File: backend/main.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import uvicorn

# --- AI Model Architecture ---
class TabularANN(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Image Branch
        self.cnn_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnn_branch.fc.in_features
        self.cnn_branch.fc = nn.Linear(num_ftrs, 128)

        # Tabular Branch
        self.ann_branch = TabularANN(input_dim=5)

        # Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img, tab):
        img_features = self.cnn_branch(img)
        tab_features = self.ann_branch(tab)
        combined = torch.cat((img_features, tab_features), dim=1)
        return self.fusion_layer(combined)

# Initialize App
app = FastAPI()
model = MultimodalFusionModel()
model.eval()

# Image Preprocessing
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_health_data(
    image: UploadFile = File(...),
    patient_data: str = Form(...)
):
    try:
        data = json.loads(patient_data)

        tab_tensor = torch.tensor([[
            float(data.get('age', 0)),
            float(data.get('bmi', 0)),
            float(data.get('systolic', 0)),
            float(data.get('diastolic', 0)),
            float(data.get('history_score', 0))
        ]], dtype=torch.float32)

        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = img_transforms(img).unsqueeze(0)

        with torch.no_grad():
            probability = model(img_tensor, tab_tensor).item()

        return {
            "probability": round(probability * 100, 2),
            "status": "High Risk" if probability > 0.5 else "Normal",
            "message": "Consult a doctor immediately." if probability > 0.5 else "All looks good."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)