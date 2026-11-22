"""Shared OCTA projection-map decoding and normalization for training and API use."""
from pathlib import Path

import numpy as np
from skimage import io, transform


def load_octa_image(path: str | Path, size: int = 256) -> np.ndarray:
    """Decode, resize, and percentile-normalize an OCTA image consistently."""
    image = io.imread(str(path), as_gray=True)
    image = np.asarray(image, dtype=np.float32)
    image = transform.resize(
        image,
        (size, size),
        anti_aliasing=True,
    ).astype(np.float32, copy=False)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError(f"Image has insufficient intensity variation: {path}")
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32, copy=False)
