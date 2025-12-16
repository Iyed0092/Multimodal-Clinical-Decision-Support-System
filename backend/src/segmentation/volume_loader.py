import os
import random
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from backend.src.segmentation.augmentations_3d import Augment3D
except ImportError:
    from src.segmentation.augmentations_3d import Augment3D


class BraTSDataset(Dataset):
    def __init__(
        self,
        root_dir,
        phase="train",
        modalities=("flair", "t1", "t1ce", "t2"),
        target_shape=(96, 128, 128),
        augment=True,
        clip_percentiles=(0.5, 99.5)
    ):
        self.root_dir = root_dir
        self.phase = phase
        self.modalities = modalities
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles

        self.use_augmentation = (augment and phase == "train")
        self.augmenter = Augment3D() if self.use_augmentation else None

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.patients = sorted(
            [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]
        )

        if not self.patients:
            raise RuntimeError(f"No patients found in {root_dir}")

        print(f"[{phase.upper()}] {len(self.patients)} patients loaded. Modalities={self.modalities}. Augment={'ON' if self.use_augmentation else 'OFF'}")

    def __len__(self):
        return len(self.patients)

    def _find_file(self, folder, key):
        pattern = os.path.join(folder, f"*{key}*.nii*")
        matches = glob.glob(pattern)
        if not matches:
            return None
        if len(matches) > 1:
            basename = os.path.basename(folder)
            preferred = [m for m in matches if basename in os.path.basename(m)]
            if preferred:
                return preferred[0]
        return matches[0]

    def _load_nifti(self, path):
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32)

    def _normalize(self, vol):
        mask = vol > 0
        if not np.any(mask):
            return vol
        low, high = np.percentile(vol[mask], self.clip_percentiles)
        vol = np.clip(vol, low, high)
        mean = vol[mask].mean()
        std = vol[mask].std()
        if std > 0:
            vol = (vol - mean) / std
        vol[~mask] = 0
        return vol

    def _process_labels(self, seg):
        wt = (seg > 0).astype(np.float32)
        tc = ((seg == 1) | (seg == 4)).astype(np.float32)
        et = (seg == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)

    def _get_crop_coords(self, mask):
        _, D, H, W = mask.shape
        td, th, tw = self.target_shape

        tumor_indices = np.argwhere(mask[0] > 0)
        center_d, center_h, center_w = D // 2, H // 2, W // 2

        if len(tumor_indices) > 0:
            if self.phase == "train":
                if random.random() < 0.8:
                    idx = random.randint(0, len(tumor_indices) - 1)
                    center_d, center_h, center_w = tumor_indices[idx]
                else:
                    center_d = random.randint(td // 2, D - td // 2)
                    center_h = random.randint(th // 2, H - th // 2)
                    center_w = random.randint(tw // 2, W - tw // 2)
            else:
                center_d = int(np.mean(tumor_indices[:, 0]))
                center_h = int(np.mean(tumor_indices[:, 1]))
                center_w = int(np.mean(tumor_indices[:, 2]))

        d0 = center_d - td // 2
        h0 = center_h - th // 2
        w0 = center_w - tw // 2

        d0 = max(0, min(d0, D - td))
        h0 = max(0, min(h0, H - th))
        w0 = max(0, min(w0, W - tw))

        return d0, h0, w0

    def _crop_volume(self, vol, d0, h0, w0):
        td, th, tw = self.target_shape
        cropped = vol[:, d0:d0 + td, h0:h0 + th, w0:w0 + tw]
        _, cd, ch, cw = cropped.shape
        if cd < td or ch < th or cw < tw:
            pad_d = td - cd
            pad_h = th - ch
            pad_w = tw - cw
            cropped = np.pad(cropped, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        return cropped

    def __getitem__(self, index):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                idx = index if attempt == 0 else random.randint(0, len(self.patients) - 1)
                patient = self.patients[idx]
                patient_dir = os.path.join(self.root_dir, patient)

                paths = {}
                for mod in self.modalities:
                    p = self._find_file(patient_dir, mod)
                    if p is None:
                        raise FileNotFoundError(f"Missing {mod} for {patient}")
                    paths[mod] = p

                seg_path = self._find_file(patient_dir, "seg")
                if seg_path is None:
                    raise FileNotFoundError(f"Missing seg for {patient}")

                images = np.stack([self._normalize(self._load_nifti(paths[m])) for m in self.modalities])
                seg = self._load_nifti(seg_path)

                targets = self._process_labels(seg)

                d0, h0, w0 = self._get_crop_coords(targets)
                images = self._crop_volume(images, d0, h0, w0)
                targets = self._crop_volume(targets, d0, h0, w0)

                if self.augmenter:
                    images, targets = self.augmenter(images, targets)

                return (
                    torch.from_numpy(np.ascontiguousarray(images)).float(),
                    torch.from_numpy(np.ascontiguousarray(targets)).float()
                )

            except Exception as e:
                print(f"Warning: failed loading {patient} (attempt {attempt + 1}/{max_retries}): {e}")

        c_in = len(self.modalities)
        c_out = 3
        td, th, tw = self.target_shape
        return torch.zeros((c_in, td, th, tw)), torch.zeros((c_out, td, th, tw))
