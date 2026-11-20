
from pathlib import Path
import pandas as pd

# BASE = Path("data")

# IMAGE_DIR = BASE / "OCTA(ILM_OPL)"
# LABEL_DIR = BASE / "Label" / "Label"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE = PROJECT_ROOT / "data"

IMAGE_DIR = BASE / "OCTA(ILM_OPL)"
LABEL_DIR = BASE / "Label" / "Label"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Image folder:", IMAGE_DIR)
print("Image folder exists:", IMAGE_DIR.exists())
print("Label folder exists:", LABEL_DIR.exists())

folders = {
    "artery": "GT_Artery",
    "vein": "GT_Vein",
    "capillary": "GT_Capillary",
    "faz": "GT_FAZ",
    "large_vessel": "GT_LargeVessel",
}

records = []

for image in IMAGE_DIR.glob("*"):
    if image.suffix.lower() not in [".bmp", ".png", ".jpg", ".tif"]:
        continue

    image_id = image.stem

    row = {
        "id": image_id,
        "image_path": str(image),
    }

    for key, folder in folders.items():
        mask = LABEL_DIR / folder / f"{image_id}.bmp"
        row[key + "_mask"] = str(mask) if mask.exists() else ""

    records.append(row)

df = pd.DataFrame(records)
Path("outputs").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DIR / "octa500_manifest.csv", index=False)
#df.to_csv("outputs/octa500_manifest.csv", index=False)

if df.empty:
    print("ERROR: No images found. Check image folder and extensions.")
    raise SystemExit(1)

print("Total images:", len(df))
print("Matched artery:", (df["artery_mask"] != "").sum())
print("Matched vein:", (df["vein_mask"] != "").sum())
print("Matched capillary:", (df["capillary_mask"] != "").sum())
print("CSV generated successfully!")
