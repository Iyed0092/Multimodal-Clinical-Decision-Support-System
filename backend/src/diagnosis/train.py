import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from backend.src.diagnosis.classifier import XRayClassifier

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
DATA_DIR = "backend/data/raw/xrays"
SAVE_PATH = "backend/data/models/best_xray_model.pth"
PLOT_PATH = "xray_training_curves.png"
PATIENCE = 5


def get_data_loaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not os.path.exists(val_dir):
        print(f"Validation directory not found: {val_dir}")
        return None, None

    train_ds = datasets.ImageFolder(train_dir, train_transform)
    val_ds = datasets.ImageFolder(val_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def save_plots(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, marker='o', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def evaluate_now(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return val_loss / len(loader), correct / total


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} (batch_size={BATCH_SIZE})")

    train_loader, val_loader = get_data_loaders()
    if not train_loader:
        return

    if os.path.exists(SAVE_PATH):
        print(f"Resuming from existing model: {SAVE_PATH}")
        classifier = XRayClassifier(weights_path=SAVE_PATH)
    else:
        print("Starting training from scratch")
        classifier = XRayClassifier()

    model = classifier.model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    print("Computing baseline performance on validation set")
    initial_loss, initial_acc = evaluate_now(model, val_loader, device, criterion)
    print(f"Baseline - Loss: {initial_loss:.4f} | Acc: {initial_acc:.2%}")

    best_loss = initial_loss
    epochs_no_improve = 0

    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        avg_val_loss, accuracy = evaluate_now(model, val_loader, device, criterion)

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(accuracy)

        save_plots(history_train_loss, history_val_loss, history_val_acc)

        scheduler.step(avg_val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={accuracy:.2%} | time={elapsed:.0f}s")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print("Saved new best model")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered")
                break


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
