import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.src.segmentation.unet_3d import UNet3D
from backend.src.segmentation.volume_loader import BraTSDataset

DATA_DIR = "backend/data/raw/brats_test"
MODEL_PATH = "backend/data/models/unet3d_brats.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

BEST_IMG = "segmentation_best_case.png"
WORST_IMG = "segmentation_worst_case.png"

LABELS = ["WT (Whole Tumor)", "TC (Tumor Core)", "ET (Enhancing Tumor)"]

def dice_score(pred, gt, eps=1e-5):
    scores = []
    for i in range(pred.shape[1]):
        p = pred[0, i].reshape(-1)
        t = gt[0, i].reshape(-1)
        d = (2*(p*t).sum() + eps)/(p.sum() + t.sum() + eps)
        scores.append(d.item())
    return scores

def save_image(x, y, pred, path, scores, patient_id):
    vol = x.cpu().numpy()[0,0]
    gt = y.cpu().numpy()[0,0]
    pr = pred.cpu().numpy()[0,0]

    if gt.sum() > 0:
        idx = np.argmax(gt.sum(axis=(1,2)))
    elif pr.sum() > 0:
        idx = np.argmax(pr.sum(axis=(1,2)))
    else:
        idx = vol.shape[0] // 2

    plt.figure(figsize=(14,5))
    title = f"WT {scores[0]:.2f} | TC {scores[1]:.2f} | ET {scores[2]:.2f}"
    plt.suptitle(f"Patient: {patient_id} | Slice {idx} | {title}")

    plt.subplot(1,3,1)
    plt.imshow(vol[idx], cmap="gray")
    plt.axis("off")
    plt.title("MRI")

    plt.subplot(1,3,2)
    plt.imshow(gt[idx], cmap="hot")
    plt.axis("off")
    plt.title("GT")

    plt.subplot(1,3,3)
    plt.imshow(pr[idx], cmap="hot")
    plt.axis("off")
    plt.title("Prediction")

    plt.tight_layout()
    try:
        plt.savefig(path)
    except:
        print("Failed to save", path)
    plt.close()

def evaluate():
    print("Running on", DEVICE)
    if not os.path.isdir(DATA_DIR):
        print("Data dir missing, aborting")
        return

    ds = BraTSDataset(DATA_DIR, phase="val")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    print("Patients:", len(ds))

    model = UNet3D(in_channels=4, n_classes=3).to(DEVICE)
    if not os.path.isfile(MODEL_PATH):
        print("Model not found")
        return
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) else ckpt)
    except Exception as e:
        print("Load failed:", e)
        return

    model.eval()
    results = []
    best_score, worst_score = 0, 1
    best_patient, worst_patient = "None", "None"
    tD, tH, tW = 64, 128, 128

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dl)):
            patient_id = ds.patients[i]
            _, _, D, H, W = x.shape
            d0, h0, w0 = max(0,(D-tD)//2), max(0,(H-tH)//2), max(0,(W-tW)//2)
            x_crop = x[:,:,d0:d0+tD, h0:h0+tH, w0:w0+tW].to(DEVICE)
            y_crop = y[:,:,d0:d0+tD, h0:h0+tH, w0:w0+tW].to(DEVICE)

            try:
                with torch.amp.autocast('cuda', enabled=DEVICE=="cuda"):
                    out = model(x_crop)
                    prob = torch.sigmoid(out)
                pred = (prob>0.5).float()
            except:
                print("Forward pass failed for", patient_id)
                continue

            scores = dice_score(pred, y_crop)
            results.append(scores)

            if scores[0] > best_score:
                best_score = scores[0]
                best_patient = patient_id
                save_image(x_crop, y_crop, pred, BEST_IMG, scores, patient_id)
            if scores[0] < worst_score:
                worst_score = scores[0]
                worst_patient = patient_id
                save_image(x_crop, y_crop, pred, WORST_IMG, scores, patient_id)

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)

    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    for i, name in enumerate(LABELS):
        print(f"{name:<22}  mean={mean[i]:.4f}  std={std[i]:.4f}")
    print("-"*40)
    print(f"Best case: {best_patient} (WT: {best_score:.4f})")
    print(f"Worst case: {worst_patient} (WT: {worst_score:.4f})")
    print("="*40)
    print("Saved in:", os.getcwd())

if __name__ == "__main__":
    evaluate()
