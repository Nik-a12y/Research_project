"""Extract baseline, vascular topology, and mask agreement features for OCTA-500."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, transform
from scipy import ndimage as ndi

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import BASE_NAMES, TOPO_NAMES, baseline_features, topology_features
from src.preprocessing import load_octa_image
from src.cnn_embeddings import FrozenResNet18
from src.vessels import segment_vessels, skeleton_of


DATASET = ROOT / "data" / "OCTA500"
OUTPUTS = ROOT / "outputs"
SIZE = 256
IMAGE_FOLDERS = (("6mm", DATASET / "OCTA_6mm"), ("3mm", DATASET / "OCTA_3mm"))
GT_DIRS = {name: DATASET / "Label" / f"GT_{name}" for name in ("Artery", "Vein", "Capillary")}


def load_gray(path: Path, size=SIZE, is_mask=False):
    if is_mask:
        image = io.imread(str(path), as_gray=True)
        image = transform.resize(image > 0, (size, size), order=0, preserve_range=True,
                                 anti_aliasing=False) > 0
        return image
    return load_octa_image(path, size=size)


def dice_score(predicted, reference):
    denominator = int(predicted.sum()) + int(reference.sum())
    return float(2 * np.logical_and(predicted, reference).sum() / denominator) if denominator else 1.0


def scaled_topology(mask, skeleton, fov_mm):
    """Express scale-sensitive features in physical units (mm and mm²)."""
    values = dict(zip(TOPO_NAMES, topology_features(mask, skeleton)))
    mm_per_pixel = fov_mm / SIZE
    values["endpoints_k"] *= 1000 / (mm_per_pixel ** 2)
    values["branchpoints_k"] *= 1000 / (mm_per_pixel ** 2)
    values["mean_comp_size"] *= mm_per_pixel ** 2
    values["mean_segment_len"] *= mm_per_pixel
    values["max_avascular_gap"] *= mm_per_pixel
    distance = ndi.distance_transform_edt(~mask) * mm_per_pixel
    values["frac_far_from_vessel"] = float((distance > 5 * 6 / SIZE).mean())
    return values


def extract_all():
    labels = pd.read_csv(ROOT / "data" / "labels.csv", dtype={"id": str})
    records = []
    cnn_images = []
    for fov, folder in IMAGE_FOLDERS:
        image_dir = folder / "Projection Maps" / "OCTA(ILM_OPL)"
        for image_path in sorted(image_dir.glob("*.bmp")):
            image_id = image_path.stem
            image = load_gray(image_path)
            cnn_images.append(image)
            mask = segment_vessels(image)
            skeleton = skeleton_of(mask)
            row = {"id": image_id, "fov": fov}
            row.update(dict(zip(BASE_NAMES, baseline_features(image, mask))))
            row.update(scaled_topology(mask, skeleton, int(fov[:-2])))
            row["skeleton_pixels"] = int(skeleton.sum())
            row["vessel_density"] = float(mask.mean())
            for vessel, gt_dir in GT_DIRS.items():
                gt_path = gt_dir / f"{image_id}.bmp"
                if gt_path.is_file():
                    gt = load_gray(gt_path, is_mask=True)
                    row[f"{vessel.lower()}_dice"] = dice_score(mask, gt)
            records.append(row)
            if len(records) % 25 == 0:
                print(f"Processed {len(records)} scans")

    features = pd.DataFrame(records)
    encoder = FrozenResNet18()
    print(f"Extracting frozen ResNet18 embeddings on {encoder.device}", flush=True)
    embeddings = encoder.embed(np.stack(cnn_images))
    embedding_frame = pd.DataFrame(
        embeddings,
        columns=[f"cnn_{index:03d}" for index in range(embeddings.shape[1])],
        index=features.index,
    )
    features = pd.concat([features, embedding_frame], axis=1)
    features = features.merge(labels, on=["id", "fov"], how="left", validate="one_to_one")
    if len(features) != 500 or features["disease"].isna().any():
        raise ValueError(f"Expected 500 labeled scans; extracted {len(features)}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(encoder.state_dict(), OUTPUTS / "resnet18_imagenet1k_v1.pth")
    features.to_csv(OUTPUTS / "topology_features_500.csv", index=False)
    # Keep the established filename used by the API and existing frontend.
    features.to_csv(OUTPUTS / "topology_features.csv", index=False)
    summary = {
        "total_images_processed": int(len(features)),
        "mean_vessel_density": float(features["vessel_density"].mean()),
        "mean_skeleton_pixels": float(features["skeleton_pixels"].mean()),
        "cnn_embedding_dim": int(embeddings.shape[1]),
        "fov_counts": features["fov"].value_counts().to_dict(),
    }
    for vessel in GT_DIRS:
        column = f"{vessel.lower()}_dice"
        if column in features:
            summary[f"mean_{column}"] = float(features[column].mean())
    with (OUTPUTS / "feature_extraction_summary.json").open("w", encoding="utf-8") as handle:
        import json
        json.dump(summary, handle, indent=2)
    plt.figure(figsize=(8, 4))
    plt.hist(features["vessel_density"] * 100, bins=25)
    plt.xlabel("Vessel density (%)")
    plt.ylabel("Subjects")
    plt.title("OCTA-500 vessel density")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "vessel_density.png", dpi=150)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.bar(TOPO_NAMES, [features[name].mean() for name in TOPO_NAMES])
    plt.xticks(rotation=55, ha="right")
    plt.title("Mean physical-scale vascular topology features")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "topology_analysis.png", dpi=150)
    plt.close()
    print(summary)
    return features


if __name__ == "__main__":
    extract_all()
