"""Synthetic OCTA-like images (DEMO ONLY - not real patient data).

Class 0 = normal, 1 = mild DR-like, 2 = severe DR-like.
Higher class -> more capillary dropout patches, larger FAZ, more broken vessels.
This only exists so the whole pipeline can be tested without the real dataset.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.draw import line

def _grow_vessels(size, rng, n_seeds=26, steps=70):
    img = np.zeros((size, size), bool)
    c = size / 2
    stack = []
    for _ in range(n_seeds):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.3, 0.45) * size
        x, y = c + r * np.cos(ang), c + r * np.sin(ang)
        d = np.arctan2(c - y, c - x) + rng.normal(0, 0.5)
        stack.append((x, y, d, steps))
    while stack:
        x, y, d, s = stack.pop()
        for _ in range(s):
            d += rng.normal(0, 0.35)
            nx, ny = x + 3 * np.cos(d), y + 3 * np.sin(d)
            if not (1 <= nx < size - 1 and 1 <= ny < size - 1):
                break
            rr, cc = line(int(y), int(x), int(ny), int(nx))
            img[rr, cc] = True
            x, y = nx, ny
            if rng.random() < 0.06 and s > 15:
                stack.append((x, y, d + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), s // 2))
    return img

def make_image(label, size=128, seed=None):
    rng = np.random.default_rng(seed)
    v = _grow_vessels(size, rng)
    v = binary_dilation(v, iterations=1)
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    faz_r = [8, 11, 15][label] * rng.uniform(0.85, 1.15)
    v[(yy - c) ** 2 + (xx - c) ** 2 < faz_r ** 2] = False
    for _ in range([2, 5, 9][label] + rng.integers(0, 2)):   # nonperfusion patches
        px, py = rng.uniform(15, size - 15, 2)
        pr = rng.uniform(6, 13)
        v[(yy - py) ** 2 + (xx - px) ** 2 < pr ** 2] = False
    img = v.astype(float)
    img = gaussian_filter(img, 0.8)
    img = img + rng.normal(0, 0.06, img.shape) + 0.04 * rng.random(img.shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32)

def make_dataset(n_per_class=60, size=128, seed=0):
    X, y = [], []
    for lab in range(3):
        for i in range(n_per_class):
            X.append(make_image(lab, size, seed=seed * 100000 + lab * 1000 + i))
            y.append(lab)
    return np.array(X), np.array(y)
