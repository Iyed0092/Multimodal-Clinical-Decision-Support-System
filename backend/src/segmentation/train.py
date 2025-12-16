import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from backend.src.segmentation.unet_3d import UNet3D
from backend.src.segmentation.volume_loader import BraTSDataset


# config
BATCH_SIZE = 1
LR = 3e-4
EPOCHS = 50
DATA_DIR = "backend/data/raw/brats_train"
SAVE_PATH = "backend/data/models/unet3d_brats.pth"
PLOT_PATH = "training_curves_segmentation.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"
SEED = 42

IN_CHANNELS = 4
NUM_CLASSES = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def dice_per_channel(prob, target, smooth=1e-6):
    dices = []
    for c in range(prob.shape[1]):
        p = prob[:, c].reshape(prob.size(0), -1)
        t = target[:, c].reshape(target.size(0), -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice = (2 * inter + smooth) / (denom + smooth)
        dices.append(dice.mean())
    return dices


def dice_loss(prob, target):
    dices = dice_per_channel(prob, target)
    return 1.0 - torch.mean(torch.stack(dices))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dice_sum = torch.zeros(NUM_CLASSES, device=DEVICE)
    count = 0

    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        probs = torch.sigmoid(logits)

        dices = dice_per_channel(probs, y)
        dice_sum += torch.stack(dices)
        count += 1

    mean_dice = dice_sum / max(1, count)
    return mean_dice.cpu().numpy()


def save_plots(loss_hist, dice_hist):
    epochs = range(1, len(loss_hist) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    dice_arr = np.array(dice_hist)
    labels = ["WT", "TC", "ET"]
    for i, lbl in enumerate(labels):
        plt.plot(epochs, dice_arr[:, i], label=lbl)
    plt.legend()
    plt.title("Validation Dice")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def train():
    print(f"Starting training on {DEVICE}")

    train_ds = BraTSDataset(DATA_DIR, phase="train")
    val_ds = BraTSDataset(DATA_DIR, phase="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = UNet3D(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

    best_mean_dice = 0.0
    start_epoch = 0

    loss_history = []
    dice_history = []

    if os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
        # expect checkpoint dict with keys: model_state, optim_state, epoch, best_dice
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt.get("optim_state", optimizer.state_dict()))
            start_epoch = ckpt.get("epoch", 0)
            best_mean_dice = ckpt.get("best_dice", 0.0)
            print(f"Resumed from checkpoint at epoch {start_epoch} (best_mean_dice={best_mean_dice:.4f})")
        else:
            print("Warning: checkpoint format unexpected; loading raw state_dict")
            model.load_state_dict(ckpt)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
                logits = model(x)
                probs = torch.sigmoid(logits)
                loss = bce(logits, y) + dice_loss(probs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)

        val_dice = evaluate(model, val_loader)
        mean_dice = val_dice.mean()
        dice_history.append(val_dice)

        scheduler.step(mean_dice)

        print(
            f"Epoch {epoch+1} | Loss={epoch_loss:.4f} | "
            f"WT={val_dice[0]:.4f} TC={val_dice[1]:.4f} ET={val_dice[2]:.4f} | mean={mean_dice:.4f}"
        )

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            print("New best model — saving checkpoint")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_dice": best_mean_dice
            }, SAVE_PATH)

        save_plots(loss_history, dice_history)

    print("Training finished.")


if __name__ == "__main__":
    train()
