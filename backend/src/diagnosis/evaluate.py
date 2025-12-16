import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from backend.src.diagnosis.classifier import XRayClassifier

DATA_DIR = "backend/data/raw/xrays/val"
MODEL_PATH = "backend/data/models/best_xray_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


def evaluate_model():
    print(f"Starting X-ray evaluation on {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Validation directory not found: {DATA_DIR}")
        return

    dataset = datasets.ImageFolder(DATA_DIR, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    print(f"Loaded {len(dataset)} images")
    print(f"Classes: {dataset.classes}")

    classifier = XRayClassifier()
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        classifier.model.load_state_dict(state)
        print("Model weights loaded")
    else:
        print("Warning: model weights not found, using untrained model")

    model = classifier.model
    model.eval()

    y_true = []
    y_scores = []
    y_pred = []

    print("Running inference")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            y_true.extend(labels.numpy())
            y_scores.extend(probs)
            y_pred.extend((probs > 0.5).astype(int))

    print("\n" + "=" * 40)
    print("FINAL DENSENET121 REPORT")
    print("=" * 40)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=dataset.classes
        )
    )

    auc_score = roc_auc_score(y_true, y_scores)
    print(f"AUC score: {auc_score:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (AUC = {auc_score:.2f})")

    output_path = "xray_evaluation_matrix.png"
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_model()
