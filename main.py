# import warnings; warnings.filterwarnings("ignore")
# """Run the full experiment:  python main.py            (synthetic demo)
#                               python main.py --real IMG_DIR LABELS.csv  (real data)"""
# import argparse, json, os, numpy as np
# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_predict
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
# from src.synthetic import make_dataset
# from src.vessels import segment_vessels, skeleton_of
# from src.features import (baseline_features, topology_features, BASE_NAMES, TOPO_NAMES)

# p = argparse.ArgumentParser()
# p.add_argument("--real", nargs=2, metavar=("IMG_DIR", "LABELS_CSV"))
# p.add_argument("--out", default="outputs")
# p.add_argument("--seed", type=int, default=0)
# a = p.parse_args()
# os.makedirs(a.out, exist_ok=True)

# # ---- Step 1: data
# if a.real:
#     from src.real_data import load_folder
#     X, y, classes, _ = load_folder(*a.real)
#     print("REAL data:", X.shape, classes)
# else:
#     X, y = make_dataset(60, 128, a.seed); classes = ["normal", "mild", "severe"]
#     print("SYNTHETIC demo data:", X.shape, "(NOT real patients)")

# # ---- Steps 2-4: segment, skeleton, features
# B, T, masks, skels = [], [], [], []
# for im in X:
#     m = segment_vessels(im); s = skeleton_of(m)
#     B.append(baseline_features(im, m)); T.append(topology_features(m, s))
#     masks.append(m); skels.append(s)
# B, T = np.array(B), np.array(T)
# BT = np.hstack([B, T])

# # ---- Step 5: evaluation
# def metrics(yt, yp):
#     return dict(accuracy=accuracy_score(yt, yp),
#                 precision_macro=precision_score(yt, yp, average="macro", zero_division=0),
#                 recall_macro=recall_score(yt, yp, average="macro", zero_division=0),
#                 f1_macro=f1_score(yt, yp, average="macro", zero_division=0))
# mk = lambda: RandomForestClassifier(300, random_state=a.seed, n_jobs=-1)
# idx = np.arange(len(y))
# tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=a.seed)
# res = {}
# for name, F in [("baseline", B), ("baseline+topology", BT)]:
#     m = mk().fit(F[tr], y[tr]); pred = m.predict(F[te])
#     res[name] = dict(heldout=metrics(y[te], pred))
#     cm = confusion_matrix(y[te], pred)
#     ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues")
#     plt.title(f"{name} (held-out test)"); plt.savefig(f"{a.out}/cm_{name.replace('+','_')}.png", dpi=120); plt.close()
#     # repeated CV macro-F1 (same folds for both models -> fair paired comparison)
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=a.seed)
#     f1s = []
#     for tri, tei in cv.split(F, y):
#         f1s.append(f1_score(y[tei], mk().fit(F[tri], y[tri]).predict(F[tei]), average="macro"))
#     res[name]["cv_f1_macro_folds"] = f1s
#     res[name]["cv_f1_macro_mean"] = float(np.mean(f1s)); res[name]["cv_f1_macro_std"] = float(np.std(f1s))
# d = np.array(res["baseline+topology"]["cv_f1_macro_folds"]) - np.array(res["baseline"]["cv_f1_macro_folds"])
# res["paired_cv_f1_diff_mean"] = float(d.mean())
# res["paired_cv_f1_diff_std"] = float(d.std())
# res["note"] = "CV folds overlap -> treat any significance test with caution; report effect size + CI."
# json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)

# # ---- feature importance for the topology model
# m = mk().fit(BT[tr], y[tr]); names = BASE_NAMES + TOPO_NAMES
# top = np.argsort(m.feature_importances_)[::-1][:12]
# plt.figure(figsize=(7, 4)); plt.barh([names[i] for i in top][::-1], m.feature_importances_[top][::-1])
# plt.title("Top features (baseline+topology model)"); plt.tight_layout()
# plt.savefig(f"{a.out}/feature_importance.png", dpi=120); plt.close()

# # ---- pipeline picture: image -> mask -> skeleton, one per class
# fig, ax = plt.subplots(3, 3, figsize=(8, 8))
# for c in range(len(classes)):
#     i = np.where(y == c)[0][0]
#     for j, (im, t) in enumerate([(X[i], "OCTA"), (masks[i], "vessel mask"), (skels[i], "skeleton")]):
#         ax[c, j].imshow(im, cmap="gray"); ax[c, j].axis("off"); ax[c, j].set_title(f"{classes[c]}: {t}", fontsize=8)
# plt.tight_layout(); plt.savefig(f"{a.out}/pipeline_examples.png", dpi=120); plt.close()

# print(json.dumps({k: (v if k.startswith("paired") else {kk: vv for kk, vv in v.items() if kk != "cv_f1_macro_folds"})
#                   for k, v in res.items() if k != "note"}, indent=2))




import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from src.real_data import load_octa_manifest
from src.vessels import segment_vessels, skeleton_of
from src.features import (
    baseline_features,
    topology_features,
    BASE_NAMES,
    TOPO_NAMES
)

# Project directories
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT / "octa500_manifest.csv"

# Load actual OCTA500 data
records = load_octa_manifest(CSV_PATH, size=256)

if not records:
    raise ValueError("No OCTA images loaded. Check manifest paths.")

feature_rows = []

# Process images
for index, record in enumerate(records):

    image = record["image"]
    image_id = record["id"]

    # Segment vessels
    predicted_mask = segment_vessels(image)
    skeleton = skeleton_of(predicted_mask)

    # Ground truth masks
    artery = record["artery_mask"]
    vein = record["vein_mask"]
    capillary = record["capillary_mask"]

    # Extract features
    base = baseline_features(image, predicted_mask)
    topo = topology_features(predicted_mask, skeleton)

    row = {
        "id": image_id,
        # Keep density as a fraction in the feature table (same as baseline_features).
        "vessel_density": float(predicted_mask.mean()),
        "skeleton_pixels": int(skeleton.sum())
    }

    for name, value in zip(BASE_NAMES, base):
        row[name] = float(value)

    for name, value in zip(TOPO_NAMES, topo):
        row[name] = float(value)

    # Ground truth comparison
    for name, gt in [
        ("artery", artery),
        ("vein", vein),
        ("capillary", capillary)
    ]:
        if gt is not None:
            intersection = np.logical_and(predicted_mask, gt).sum()
            union = np.logical_or(predicted_mask, gt).sum()

            row[name + "_dice"] = float(
                2 * intersection /
                (predicted_mask.sum() + gt.sum() + 1e-9)
            )

            row[name + "_iou"] = float(
                intersection / (union + 1e-9)
            )

    feature_rows.append(row)

    # Save example visualizations
    if index < 10:
        fig, ax = plt.subplots(1, 6, figsize=(18, 4))

        images = [
            image,
            predicted_mask,
            skeleton,
            artery if artery is not None else np.zeros_like(image),
            vein if vein is not None else np.zeros_like(image),
            capillary if capillary is not None else np.zeros_like(image)
        ]

        titles = [
            "Original OCTA",
            "Predicted vessels",
            "Skeleton",
            "GT Artery",
            "GT Vein",
            "GT Capillary"
        ]

        for a, im, title in zip(ax, images, titles):
            a.imshow(im, cmap="gray")
            a.set_title(title)
            a.axis("off")

        plt.tight_layout()
        plt.savefig(
            OUT / f"comparison_{image_id}.png",
            dpi=150
        )
        plt.close()

    if (index + 1) % 25 == 0:
        print(f"Processed {index + 1}/{len(records)} images")

# Save feature results
df = pd.DataFrame(feature_rows)
df.to_csv(OUT / "topology_features.csv", index=False)

# Summary
summary = {
    "total_images_processed": len(df),
    "mean_vessel_density": float(df["vessel_density"].mean()),
    "mean_skeleton_pixels": float(df["skeleton_pixels"].mean()),
}

with open(OUT / "results.json", "w") as f:
    json.dump(summary, f, indent=4)

# Vessel density chart
plt.figure(figsize=(9, 5))
plt.hist(df["vessel_density"] * 100, bins=20)
plt.xlabel("Vessel density (%)")
plt.ylabel("Number of images")
plt.title("OCTA500 Vessel Density Distribution")
plt.tight_layout()
plt.savefig(OUT / "vessel_density.png", dpi=150)
plt.close()

# Topology feature chart
plt.figure(figsize=(10, 5))
plt.bar(
    TOPO_NAMES,
    [df[name].mean() for name in TOPO_NAMES]
)
plt.xticks(rotation=60, ha="right")
plt.title("Average Topological Features")
plt.tight_layout()
plt.savefig(OUT / "topology_analysis.png", dpi=150)
plt.close()

print("\nPROCESSING COMPLETED!")
print(json.dumps(summary, indent=4))
print("Results saved in:", OUT)
