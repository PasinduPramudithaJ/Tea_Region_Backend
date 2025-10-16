import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import io

# Flask setup
app = Flask(__name__)
CORS(app)

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Region info
region_info = {
    "Sabaragamuwa Region": {"description": "Known for strong aroma and dark color.", "origin": "Sabaragamuwa province", "flavorNotes": ["Malty", "Earthy", "Rich"]},
    "Dimbula Region": {"description": "Balanced flavor, bright color.", "origin": "Central highlands", "flavorNotes": ["Floral", "Light", "Aromatic"]},
    "Ruhuna Region": {"description": "Smooth taste, golden color.", "origin": "Southern lowlands", "flavorNotes": ["Sweet", "Mellow", "Smooth"]}
}

# Device
device = torch.device("cpu")  # Force CPU for Render free tier

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/tea_4_region_model_.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction ---
def predict(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    return pred_class, prob_dict

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        # Read image directly from memory
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize aggressively to reduce memory usage
        img_bgr = cv2.resize(img_bgr, (224, 224))

        # Convert to RGB PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pred_class, prob_dict = predict(pil_img)
        info = region_info.get(pred_class, {})

        # Encode image to base64
        _, buffer = cv2.imencode(".png", img_bgr)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "prediction": pred_class,
            "confidence": prob_dict[pred_class],
            "probabilities": prob_dict,
            "info": info,
            "processedImage": f"data:image/png;base64,{img_b64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
