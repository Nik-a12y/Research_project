# """Step 2: vessel segmentation + skeleton."""
# import numpy as np
# from skimage.filters import sato, threshold_otsu, gaussian
# from skimage.morphology import skeletonize, remove_small_objects

# def segment_vessels(img, min_size=20):
#     """Ridge filter (Sato) -> Otsu threshold -> remove tiny specks."""
#     img = gaussian(img, 0.5)
#     ridge = sato(img, sigmas=[1, 1.5, 2], black_ridges=False)
#     mask = ridge > threshold_otsu(ridge)
#     return remove_small_objects(mask, min_size)

# def skeleton_of(mask):
#     return skeletonize(mask)



"""OCTA500: Vessel segmentation and skeletonization."""

import numpy as np
from skimage.filters import sato, threshold_otsu, gaussian
from skimage.morphology import skeletonize, remove_small_objects


def normalize_image(img):
    """Normalize OCTA image to range 0-1."""

    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)

    low, high = np.percentile(img, [1, 99])

    if high <= low:
        raise ValueError("Image has insufficient intensity variation.")

    img = np.clip((img - low) / (high - low), 0, 1)

    return img


def segment_vessels(img, min_size=20):
    """Detect bright retinal vessels using Sato filter."""

    img = normalize_image(img)

    smooth = gaussian(img, sigma=0.5)

    ridge = sato(
        smooth,
        sigmas=[1, 1.5, 2, 3],
        black_ridges=False
    )

    if not np.isfinite(ridge).all():
        raise ValueError("Invalid values in ridge response.")

    threshold = threshold_otsu(ridge)

    mask = ridge > threshold

    # `max_size` removes objects at or below the limit; subtract one to retain
    # the historical `min_size` behavior (remove objects strictly smaller).
    mask = remove_small_objects(
        mask.astype(bool),
        max_size=min_size - 1
    )

    return mask.astype(bool)


def skeleton_of(mask):
    """Convert segmented vessels into a thin skeleton."""

    mask = np.asarray(mask, dtype=bool)

    return skeletonize(mask)


def vessel_density(mask):
    """Calculate percentage of vessel pixels."""

    mask = np.asarray(mask, dtype=bool)

    return float(np.mean(mask) * 100)


def topology_counts(skeleton):
    """Count endpoints and junction pixels."""

    skeleton = np.asarray(skeleton, dtype=bool)

    padded = np.pad(
        skeleton.astype(np.uint8),
        1
    )

    neighbors = np.zeros_like(skeleton, dtype=np.uint8)

    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue

            neighbors += padded[
                dy:dy + skeleton.shape[0],
                dx:dx + skeleton.shape[1]
            ]

    endpoints = int(np.sum(skeleton & (neighbors == 1)))
    junctions = int(np.sum(skeleton & (neighbors >= 3)))

    return {
        "endpoints": endpoints,
        "junction_pixels": junctions
    }
