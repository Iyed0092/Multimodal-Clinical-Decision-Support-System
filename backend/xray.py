import os
import base64
from io import BytesIO
import torch
import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image
from backend.src.diagnosis.classifier import XRayClassifier
from backend.src.diagnosis.gradcam import GradCAM, overlay_heatmap

xray_bp = Blueprint("xray", __name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

xray_model = None
_load_err = None
print("Loading X-Ray model...")
try:
    xray_model = XRayClassifier(weights_path="data/models/best_xray_model.pth")
    xray_model.model.eval()
    print("X-Ray ready")
except Exception as e:
    _load_err = str(e)
    print("Load failed:", e)


def img_to_b64(arr):
    try:
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
        b = BytesIO()
        im.save(b, format="JPEG", quality=90)
        return base64.b64encode(b.getvalue()).decode()
    except:
        return None


@xray_bp.route("/health")
def health():
    ok = xray_model is not None
    return jsonify({"ok": ok, "loaded": ok, "error": None if ok else _load_err})


@xray_bp.route("", methods=["POST"])
def predict_xray():
    if not xray_model:
        return jsonify({"error": "model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    path = os.path.join(UPLOAD_DIR, f.filename)
    try:
        f.save(path)
    except:
        return jsonify({"error": "save failed"}), 500

    label = None
    confidence = None
    try:
        label, confidence = xray_model.predict(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    heat_b64 = None
    try:
        cam = GradCAM(xray_model.model, xray_model.model.features.norm5)
        img_tensor = xray_model.preprocess(path)
        if img_tensor is not None:
            x = img_tensor.unsqueeze(0).to(DEVICE)
            x.requires_grad_()
            heat = cam.generate(x)
            overlay = overlay_heatmap(heat, path)
            if overlay is not None:
                heat_b64 = img_to_b64(overlay)
    except Exception as e:
        print("GradCAM fail:", e)

    try:
        os.remove(path)
    except:
        pass

    return jsonify({
        "label": label,
        "confidence": float(confidence) if confidence else None,
        "heatmap_image": heat_b64
    })
