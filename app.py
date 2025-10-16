import os
import socket
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

# Flask setup
app = Flask(__name__)
CORS(app)

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Region info for extra user-friendly data
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

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("models/tea_4_region_model_restnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Robust circle detection + reflection removal ---
def find_tea_circle(img):
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
            cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
            r = max(5, min(r, cx, cy, w - cx - 1, h - cy - 1))
            return cx, cy, r

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w)//4,
                               param1=80, param2=40, minRadius=min(h, w)//8, maxRadius=0)
    if circles is not None:
        x, y, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        r = max(5, min(r, x, y, w - x - 1, h - y - 1))
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
    """Return square crop around circle as RGBA (transparent outside the tea).
       Fill inner circle using pixel pattern from outer ring for natural gradient.
    """
    h, w = img.shape[:2]

    # main mask
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), int(0.80 * r), 255, -1)  # Outer Circle

    inner_radius = int(r * 0.42)

    # --- extract outer ring pixels ---
    ring_mask = np.zeros((h, w), np.uint8)
    cv2.circle(ring_mask, (x, y), inner_radius + 15, 255, -1)
    cv2.circle(ring_mask, (x, y), inner_radius, 0, -1)  # only outer ring
    ring_mask = cv2.bitwise_and(ring_mask, mask)

    ring_pixels = img[ring_mask == 255]

    if len(ring_pixels) > 0:
        # randomly sample pixels to fill inner circle
        inner_mask = np.zeros((h, w), np.uint8)
        cv2.circle(inner_mask, (x, y), inner_radius, 255, -1)
        indices = np.argwhere(inner_mask == 255)
        for idx in indices:
            y_idx, x_idx = idx
            img[y_idx, x_idx] = ring_pixels[np.random.randint(len(ring_pixels))]
    else:
        # fallback mean color
        mean_color = [128, 90, 60]
        cv2.circle(img, (x, y), inner_radius, mean_color, -1)

    # reflection cleanup
    cleaned = remove_reflection(img, mask)

    # RGBA output
    b, g, rch = cv2.split(cleaned)
    alpha = mask
    rgba = cv2.merge([b, g, rch, alpha])

    # crop
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)
    return rgba[y1:y2, x1:x2]

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
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        img_bgr = cv2.imread(file_path)
        circle = find_tea_circle(img_bgr)
        if circle is None:
            return jsonify({"error": "No tea circle detected"}), 400
        x, y, r = circle
        cropped = crop_circle_png(img_bgr, x, y, r)

        # Convert to PIL for model
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        pred_class, prob_dict = predict(pil_img)
        info = region_info.get(pred_class, {})

        # Encode cropped image to base64
        _, buffer = cv2.imencode(".png", cropped)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "prediction": pred_class,
            "confidence": prob_dict[pred_class],
            "probabilities": prob_dict,
            "info": info,
            "croppedImage": f"data:image/png;base64,{img_b64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Server running on:")
    print(f"  Localhost: http://127.0.0.1:5000")
    print(f"  Network:   http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
