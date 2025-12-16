import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CLASSES = ["Normal", "Pneumonia"]


class XRayClassifier:
    def __init__(self, weights_path=None):
        self.model = self._build_model().to(DEVICE)
        self.model.eval()

        self.model.forward = types.MethodType(self._safe_forward, self.model)

        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                print(f"Loaded X-ray weights from {weights_path}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print("Using default ImageNet weights")

    def _safe_forward(self, model, x):
        x = model.features(x)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return model.classifier(x)

    def _build_model(self):
        print("Initializing DenseNet121")
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        for p in model.features.parameters():
            p.requires_grad = False

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        return model

    def preprocess(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        try:
            img = Image.open(img_path).convert("RGB")
            return transform(img)
        except Exception as e:
            print(f"Preprocess error: {e}")
            return None

    def predict(self, img_path):
        img = self.preprocess(img_path)
        if img is None:
            return None, 0.0

        img = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(img)
            prob = torch.sigmoid(logits).item()

        if prob > 0.5:
            return CLASSES[1], prob
        return CLASSES[0], 1 - prob
