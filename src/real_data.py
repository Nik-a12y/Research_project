# """Load REAL OCT-A en-face images (e.g. OCTA-500).

# You MUST check the real folder/label layout after downloading and adjust here.
# Generic design: a folder of images + a CSV with columns  id,label
# (label can be text such as NORMAL / DR / AMD ...).
# Make that CSV yourself from OCTA-500's label file.
# """
# import os, numpy as np, pandas as pd
# from skimage import io, transform, img_as_float

# def load_folder(img_dir, labels_csv, size=256, keep_labels=None):
#     df = pd.read_csv(labels_csv)
#     X, y, ids = [], [], []
#     for _, r in df.iterrows():
#         if keep_labels and r["label"] not in keep_labels:
#             continue
#         found = [f for f in os.listdir(img_dir) if os.path.splitext(f)[0] == str(r["id"])]
#         if not found:
#             continue
#         im = img_as_float(io.imread(os.path.join(img_dir, found[0]), as_gray=True))
#         im = transform.resize(im, (size, size), anti_aliasing=True)
#         im = (im - im.min()) / (im.max() - im.min() + 1e-9)
#         X.append(im.astype(np.float32)); y.append(r["label"]); ids.append(r["id"])
#     classes = sorted(set(y))
#     y = np.array([classes.index(v) for v in y])
#     return np.array(X), y, classes, ids




"""OCTA500 real image and mask loader."""

from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_float


def read_image(path, size=256, is_mask=False):

    if path is None or pd.isna(path) or str(path).strip() == "":
        return None

    path = Path(path)

    if not path.exists():
        return None

    img = io.imread(str(path), as_gray=True)

    if is_mask:
        img = (img > 0).astype(np.uint8)
        img = transform.resize(
            img, (size, size),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )
        return img > 0

    img = img_as_float(img)

    img = transform.resize(
        img, (size, size),
        anti_aliasing=True
    )

    low, high = np.percentile(img, [1, 99])

    if high > low:
        img = np.clip((img - low) / (high - low), 0, 1)
    else:
        img = np.zeros_like(img)

    return img.astype(np.float32)


def load_octa_manifest(csv_path, size=256, limit=None):

    df = pd.read_csv(csv_path)

    required = [
        "id", "image_path",
        "artery_mask", "vein_mask", "capillary_mask"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if limit:
        df = df.head(limit)

    records = []

    for _, row in df.iterrows():

        image = read_image(row["image_path"], size)

        if image is None:
            continue

        record = {
            "id": str(row["id"]),
            "image": image
        }

        for name in [
            "artery_mask",
            "vein_mask",
            "capillary_mask",
            "faz_mask",
            "large_vessel_mask"
        ]:
            record[name] = read_image(
                row.get(name, ""),
                size,
                is_mask=True
            )

        records.append(record)

    print("Successfully loaded:", len(records))

    return records
