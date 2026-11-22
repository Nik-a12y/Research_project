# Topology-Aware OCT-A Retinal Analysis

MCA Semester 4 research project evaluating whether frozen ImageNet ResNet18 embeddings and explicit retinal vascular topology improve diabetic retinopathy classification from OCT-A scans. The software is for research use and is not a clinical diagnostic tool.

## Prepare and run the OCTA-500 experiment

Install Python requirements, then run these steps from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_labels.py
python scripts/extract_features.py
python scripts/train_classifier.py
```

The scripts read the official 6 mm and 3 mm workbooks and projection maps under `data/OCTA500/`. They write the unified `data/labels.csv`, 512-dimensional frozen CNN embeddings and physical-scale vascular features, cross-validated classifier results and plots, and `outputs/dr_classifier.pkl`.

Run the API with `uvicorn api:app --reload`. Start the React app from `frontend/` with `npm run dev`.

## Research design

- **CNN Baseline:** 512-dimensional ResNet18 ImageNet embeddings reduced to 32 PCA components.
- **Topology Only:** per-FOV-normalized vascular topology features.
- **Hybrid CNN + Topology:** the 32 PCA components plus per-FOV-normalized topology features.
- Binary task: NORMAL vs DR; all other diagnoses are excluded.
- Multiclass task: NORMAL vs DR vs OTHER.
- Both tasks use nested cross-validation with five stratified outer folds and four inner folds, stratified jointly by FOV and class. Each fold fits its own CNN scaler and PCA to prevent leakage. Features are z-scored within each FOV using training-fold statistics, and class probabilities are sigmoid-calibrated separately for 3 mm and 6 mm scans. DR decision thresholds are tuned per FOV to maximize F2, prioritizing sensitivity. Age and sex are retained only as label-table metadata and never enter the models.

The 3 mm and 6 mm scans are normalized to 256 x 256 before the same torchvision ImageNet preprocessing is applied in batch extraction and live inference. Scale-sensitive topology measurements are converted to physical units using field-of-view metadata. Segmentation agreement is measured against available artery, vein, and capillary masks.
