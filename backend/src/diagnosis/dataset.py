import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, phase="train"):
        self.image_paths = image_paths
        self.labels = labels
        self.phase = phase

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if phase == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            target = torch.tensor(label, dtype=torch.float32)
            return img, target
        except Exception:
            print(f"Warning: failed to load image {img_path}")
            return (
                torch.zeros(3, 224, 224),
                torch.tensor(label, dtype=torch.float32)
            )
