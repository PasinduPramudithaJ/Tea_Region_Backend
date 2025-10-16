import os
import socket
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

# --- Load class names ---
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

# --- Region info ---
region_info = {
    "Sabaragamuwa Region": {
        "description": "Known for strong aroma and dark color.",
        "origin": "Sabaragamuwa province",
        "flavorNotes": ["Malty", "Earthy", "Rich"]
    },
    "Dimbula Region": {
        "description": "Balanced flavor, bright color.",
        "origin": "Central highlands",
        "flavorNotes": ["Floral", "Light", "Aromatic"]
    },
    "Ruhuna Region": {
        "description": "Smooth taste, golden color.",
        "origin": "Southern lowlands",
        "flavorNotes": ["Sweet", "Mellow", "Smooth"]
    }
}

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/tea_4_region_model_restnet18.pth", map_location=device))
model = model.to(device).eval()

# --- Transform for model input ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict(pil_img):
    img_t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    idx = probs.argmax()
    return class_names[idx], {cls: float(probs[i]) for i, cls in enumerate(class_names)}

# --- API endpoint ---
@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Process file in-memory
        img_bytes = BytesIO(file.read())
        pil_img = Image.open(img_bytes).convert("RGB")

        # Make prediction
        pred_class, prob_dict = predict(pil_img)
        info = region_info.get(pred_class, {})

        # Encode original image for frontend
        img_bytes.seek(0)
        img_b64 = base64.b64encode(img_bytes.read()).decode("utf-8")

        return jsonify({
            "prediction": pred_class,
            "confidence": prob_dict[pred_class],
            "probabilities": prob_dict,
            "info": info,
            "image": f"data:image/jpeg;base64,{img_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Local IP helper ---
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
    ip = get_local_ip()
    print(f"Server running:\n  Local:   http://127.0.0.1:5000\n  Network: http://{ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
