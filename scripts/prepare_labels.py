"""Build the unified OCTA-500 clinical label table from the official workbooks."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "OCTA500"
OUTPUT = ROOT / "data" / "labels.csv"
SOURCES = [
    (DATASET / "OCTA_6mm" / "Text labels.xlsx", "6mm", 10001, 10300),
    (DATASET / "OCTA_3mm" / "Text labels.xlsx", "3mm", 10301, 10500),
]
OTHER_DISEASES = {"AMD", "CNV", "CSC", "RVO", "OTHERS"}


def prepare_labels() -> pd.DataFrame:
    frames = []
    for workbook, fov, first_id, last_id in SOURCES:
        if not workbook.is_file():
            raise FileNotFoundError(f"Clinical label workbook not found: {workbook}")
        frame = pd.read_excel(workbook)
        required = {"ID", "Sex", "OS/OD", "Age", "Disease"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{workbook.name} is missing columns: {sorted(missing)}")
        frame = frame.rename(columns={
            "ID": "id", "Sex": "sex", "OS/OD": "eye", "Age": "age", "Disease": "disease"
        })
        frame["id"] = frame["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["disease"] = frame["disease"].astype(str).str.strip().str.upper()
        frame = frame[frame["id"].astype(int).between(first_id, last_id)].copy()
        frame["fov"] = fov
        frame["binary_label"] = frame["disease"].map({"NORMAL": 0, "DR": 1})
        frame["multiclass_label"] = frame["disease"].map(
            {"NORMAL": 0, "DR": 1, **{label: 2 for label in OTHER_DISEASES}}
        )
        if frame["multiclass_label"].isna().any():
            unknown = sorted(frame.loc[frame["multiclass_label"].isna(), "disease"].unique())
            raise ValueError(f"Unexpected disease labels in {workbook.name}: {unknown}")
        frames.append(frame[[
            "id", "fov", "sex", "eye", "age", "disease", "binary_label", "multiclass_label"
        ]])

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.drop_duplicates("id", keep=False).sort_values("id").reset_index(drop=True)
    if len(labels) != 500 or labels["id"].nunique() != 500:
        raise ValueError(f"Expected 500 unique subjects; found {len(labels)} rows")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT, index=False)
    return labels


if __name__ == "__main__":
    result = prepare_labels()
    print(f"Wrote {len(result)} labels to {OUTPUT}")
    print(result["disease"].value_counts().sort_index().to_string())
