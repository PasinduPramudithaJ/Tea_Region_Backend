import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ResNet4 definition ---
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Load models ---
models_dict = {}

# ResNet18
resnet18_model = models.resnet18(weights=None)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, NUM_CLASSES)
resnet18_model.load_state_dict(torch.load("models/tea_4_region_model_restnet18.pth", map_location=device))
resnet18_model.to(device).eval()
models_dict["tea_4_region_model_restnet18"] = resnet18_model

# ResNet4
resnet4_model = ResNet4(NUM_CLASSES)
resnet4_model.load_state_dict(torch.load("models/tea_4_region_model_restnet4.pth", map_location=device))
resnet4_model.to(device).eval()
models_dict["tea_4_region_model_restnet4"] = resnet4_model

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Region Info ---
region_info = {
    "Sabaragamuwa Region": {"description": "Strong aroma, dark color", "origin": "Sabaragamuwa", "flavorNotes": ["Malty","Earthy","Rich"]},
    "Dimbula Region": {"description": "Balanced flavor, bright color", "origin": "Central Highlands", "flavorNotes": ["Floral","Light","Aromatic"]},
    "Ruhuna Region": {"description": "Smooth taste, golden color", "origin": "Southern lowlands", "flavorNotes": ["Sweet","Mellow","Smooth"]},
    "Nuwara Eliya Region": {"description": "Light, crisp flavor with floral notes", "origin": "Nuwara Eliya", "flavorNotes": ["Floral","Citrus","Bright"]}
}

# ======================================================
# ============== Image Preprocessing ===================
# ======================================================

def find_tea_circle(img):
    """Detect circular tea region using LAB + Hough transform."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = cv2.GaussianBlur(lab[:, :, 2], (9, 9), 0)
    _, mask = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 0.01 * h * w:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            return int(cx), int(cy), int(r)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//4,
                               param1=80, param2=40, minRadius=min(h, w)//8, maxRadius=0)
    if circles is not None:
        x, y, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        return x, y, r
    return None

def remove_reflection(img, circle_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    bright = cv2.inRange(V, 220, 255)
    low_sat = cv2.inRange(S, 0, 60)
    spec = cv2.bitwise_and(bright, low_sat)
    spec = cv2.bitwise_and(spec, circle_mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, k, 1)
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, k, 1)
    if np.count_nonzero(spec) == 0:
        return img
    return cv2.inpaint(img, spec, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def crop_circle_png(img, x, y, r):
    """Crop tea region as circular RGBA patch."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), int(0.85 * r), 255, -1)
    cleaned = remove_reflection(img, mask)
    b, g, rch = cv2.split(cleaned)
    alpha = mask
    rgba = cv2.merge([b, g, rch, alpha])
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)
    return rgba[y1:y2, x1:x2]

# ======================================================
# ============== Prediction Function ===================
# ======================================================
def predict_image(pil_img, model):
    img_t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    idx = probs.argmax()
    return class_names[idx], {cls: float(probs[i]) for i, cls in enumerate(class_names)}

# ======================================================
# ============== Flask Endpoint ========================
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files.get("file")
    model_name = request.args.get("model", "tea_4_region_model_restnet18")
    image_type = request.args.get("type", "raw")  # raw or preprocessed
    model = models_dict.get(model_name)

    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if not model:
        return jsonify({"error": f"Model '{model_name}' not found"}), 400

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        if image_type == "raw":
            circle = find_tea_circle(img_bgr)
            if circle is None:
                return jsonify({"error": "No tea circle detected"}), 400
            x, y, r = circle
            cropped = crop_circle_png(img_bgr, x, y, r)
            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGB)
        else:
            # Preprocessed type: assume already cropped
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        pred_class, prob_dict = predict_image(pil_img, model)
        confidence = prob_dict[pred_class]
        info = region_info.get(pred_class, {})

        # Encode image to base64
        _, buffer = cv2.imencode(".png", img_bgr)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "info": info,
            "processedType": image_type,
            "image": f"data:image/png;base64,{img_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ======================================================
# ============== Server Setup ==========================
# ======================================================
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

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Server running on:\n  Localhost: http://127.0.0.1:5000\n  Network: http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
