import numpy as np
import random
from scipy.ndimage import gaussian_filter


class Augment3D:
    def __init__(
        self,
        p_flip=0.5,
        p_rotate=0.5,
        p_intensity=0.3,
        p_contrast=0.3,
        p_gamma=0.3,
        p_blur=0.2,
        p_noise=0.2,
        intensity_std=0.1,
        contrast_range=(0.8, 1.2),
        gamma_range=(0.8, 1.2),
        blur_sigma=(0.5, 1.0),
        noise_std=0.05
    ):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_intensity = p_intensity
        self.p_contrast = p_contrast
        self.p_gamma = p_gamma
        self.p_blur = p_blur
        self.p_noise = p_noise

        self.intensity_std = intensity_std
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std

    def random_flip(self, x, y):
        if random.random() < self.p_flip:
            axis = random.choice([2, 3])
            x = np.flip(x, axis=axis).copy()
            y = np.flip(y, axis=axis).copy()
        return x, y

    def random_rotate_90(self, x, y):
        if random.random() < self.p_rotate:
            k = random.randint(1, 3)
            x = np.rot90(x, k, axes=(2, 3)).copy()
            y = np.rot90(y, k, axes=(2, 3)).copy()
        return x, y

    def random_bias(self, x):
        if random.random() < self.p_intensity:
            x = x + np.random.normal(0, self.intensity_std)
        return x

    def random_contrast(self, x):
        if random.random() < self.p_contrast:
            factor = random.uniform(*self.contrast_range)
            mean = x.mean()
            x = (x - mean) * factor + mean
        return x

    def random_noise(self, x):
        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
            x = x + noise
        return x

    def gamma_correction(self, x):
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            min_val = x.min()
            if min_val < 0:
                x = x - min_val
            x = np.power(x, gamma)
            if min_val < 0:
                x = x + min_val
        return x

    def gaussian_blur(self, x):
        if random.random() < self.p_blur:
            sigma = random.uniform(*self.blur_sigma)
            for c in range(x.shape[0]):
                x[c] = gaussian_filter(x[c], sigma=sigma)
        return x

    def __call__(self, x, y):
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        x, y = self.random_flip(x, y)
        x, y = self.random_rotate_90(x, y)

        x = self.random_noise(x)
        x = self.random_bias(x)
        x = self.random_contrast(x)
        x = self.gamma_correction(x)
        x = self.gaussian_blur(x)

        return x, y
