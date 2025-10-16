import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

# --- Load class names ---
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

NUM_CLASSES = len(class_names)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ResNet-4 definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class ResNet4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Load trained model ---
MODEL_PATH = "models/tea_region_model_resnet4.pth"
model = ResNet4(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    confidence = prob_dict[pred_class]
    return pred_class, confidence, prob_dict

# --- API endpoint ---
@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pred_class, confidence, prob_dict = predict(pil_img)

        # Encode original image as base64
        _, buffer = cv2.imencode(".png", img_bgr)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "image": f"data:image/png;base64,{img_b64}"
        })
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Helper: get local IP ---
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# --- Run server ---
if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Server running on:")
    print(f"  Localhost: http://127.0.0.1:5000")
    print(f"  Network:   http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
