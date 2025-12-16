import os
import uuid
import base64
from io import BytesIO

import numpy as np
import nibabel as nib
import torch
from PIL import Image
from flask import Blueprint, request, jsonify

from backend.src.segmentation.unet_3d import UNet3D

mri_bp = Blueprint("mri", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "backend/data/models/unet3d_brats.pth"

unet_model = UNet3D(in_channels=4, n_classes=3).to(DEVICE)
try:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        unet_model.load_state_dict(ckpt["model_state"])
    else:
        unet_model.load_state_dict(ckpt)
    unet_model.eval()
    print("UNet model loaded")
except Exception as e:
    print("Failed loading UNet:", e)

def image_to_base64(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_upload(f, tag):
    name = f.filename or f"{tag}.nii"
    ext = ".nii.gz" if name.endswith(".gz") else ".nii"
    path = os.path.join(UPLOAD_FOLDER, f"{tag}_{uuid.uuid4().hex}{ext}")
    f.save(path)
    return path

def load_nifti(path):
    try:
        return nib.load(path).get_fdata().astype(np.float32)
    except:
        alt = path + ".gz" if not path.endswith(".gz") else path[:-3]
        os.rename(path, alt)
        return nib.load(alt).get_fdata().astype(np.float32)


@mri_bp.route("", methods=["POST"])
def predict_mri():
    files = request.files
    main_file = files.get("file_mri") or files.get("file")
    keys = ["flair", "t1", "t1ce", "t2"]
    modal_files = {k: files.get(f"file_{k}") for k in keys}

    if not main_file and not any(modal_files.values()):
        return jsonify({"error": "no MRI uploaded"}), 400

    tmp_paths = []
    seg_vol = None

    seg_file = files.get("file_seg")
    if seg_file:
        try:
            seg_path = save_upload(seg_file, "seg")
            tmp_paths.append(seg_path)
            seg_vol = (load_nifti(seg_path) > 0).astype(np.float32)
        except:
            seg_vol = None

    def read_modal(f, tag):
        try:
            p = save_upload(f, tag)
            tmp_paths.append(p)
            return load_nifti(p)
        except:
            return None

    vols = []
    if any(modal_files.values()):
        for k in keys:
            f = modal_files[k]
            vols.append(read_modal(f, k) if f else None)
        base = next((v for v in vols if v is not None), None)
        vols = [v if v is not None else base.copy() for v in vols]
    else:
        v = read_modal(main_file, "mri")
        vols = [v.copy() for _ in range(4)]

    for i in range(4):
        mn, mx = vols[i].min(), vols[i].max()
        vols[i] = (vols[i] - mn)/(mx - mn) if mx > mn else np.zeros_like(vols[i])

    D, H, W = vols[0].shape
    tD, tH, tW = 64, 128, 128
    pad = (max(0, tD-D), max(0, tH-H), max(0, tW-W))
    if any(pad):
        vols = [np.pad(v, ((0, pad[0]), (0, pad[1]), (0, pad[2]))) for v in vols]
        if seg_vol is not None:
            seg_vol = np.pad(seg_vol, ((0, pad[0]), (0, pad[1]), (0, pad[2])))

    d0, h0, w0 = (vols[0].shape[0]-tD)//2, (vols[0].shape[1]-tH)//2, (vols[0].shape[2]-tW)//2
    stack = np.stack([v[d0:d0+tD, h0:h0+tH, w0:w0+tW] for v in vols], axis=0).astype(np.float32)

    if seg_vol is not None:
        seg_vol = seg_vol[d0:d0+tD, h0:h0+tH, w0:w0+tW]

    x = torch.from_numpy(stack).unsqueeze(0).to(DEVICE)
    try:
        probs = torch.sigmoid(unet_model(x))
    except:
        probs = torch.zeros(1,4,64,128,128)

    pred = (probs > 0.5).float().cpu().numpy()[0][0]

    areas = seg_vol.sum(axis=(1,2)) if seg_vol is not None else pred.sum(axis=(1,2))
    idx = int(np.argmax(areas)) if areas.max() > 0 else tD//2

    mri_slice = (stack[0][idx]*255).astype(np.uint8)
    pred_slice = (pred[idx]*255).astype(np.uint8)
    mri_b64 = image_to_base64(mri_slice)
    pred_b64 = image_to_base64(pred_slice)
    gt_b64 = None
    if seg_vol is not None:
        gt_b64 = image_to_base64((seg_vol[idx]*255).astype(np.uint8))

    for p in tmp_paths:
        try: os.remove(p)
        except: pass

    return jsonify({
        "slice_index": idx,
        "tumor_detected": bool(areas.max() > 0),
        "has_gt": seg_vol is not None,
        "mri_image": mri_b64,
        "segmentation_image": pred_b64,
        "gt_image": gt_b64
    })
