"""Nested five-fold CNN/topology ablation with fold-fitted PCA and calibration."""
from pathlib import Path
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, fbeta_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn_embeddings import EMBEDDING_DIM, WEIGHT_ENUM
from src.classification import (apply_fov_calibrators, dr_threshold,
                                fit_fov_calibrators)
from src.hybrid import MODEL_VARIANTS, fit_variant_transformer, transform_variant


OUTPUTS = ROOT / "outputs"
CLASS_NAMES = {"binary": ["NORMAL", "DR"], "multiclass": ["NORMAL", "DR", "OTHER"]}
CNN_CONFIG = {
    "architecture": "torchvision.models.resnet18",
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "embedding_dim": EMBEDDING_DIM,
    "classifier_head": "removed; use average-pool output",
    "embedding_preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
    "input": "grayscale OCTA repeated to RGB after shared 256x256 percentile normalization",
}


def classifier():
    return ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )


def predict_at_threshold(probabilities, classes, threshold, task):
    classes = np.asarray(classes, dtype=int)
    dr_column = int(np.flatnonzero(classes == 1)[0])
    dr_rows = probabilities[:, dr_column] >= threshold
    if task == "binary":
        return np.where(dr_rows, 1, 0)
    non_dr_columns = [i for i, label in enumerate(classes) if label != 1]
    non_dr = classes[non_dr_columns][np.argmax(probabilities[:, non_dr_columns], axis=1)]
    return np.where(dr_rows, 1, non_dr)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "dr_precision": float(precision_score(y_true == 1, y_pred == 1, zero_division=0)),
        "dr_recall": float(recall_score(y_true == 1, y_pred == 1, zero_division=0)),
        "dr_f2": float(fbeta_score(y_true == 1, y_pred == 1, beta=2, zero_division=0)),
    }


def joint_strata(frame, target):
    return frame["fov"].astype(str) + "_" + frame[target].astype(str)


def evaluate_variant(frame, target, task, variant):
    """Evaluate one ablation with fold-local FOV statistics, scaler, and PCA."""
    frame = frame.reset_index(drop=True)
    y = frame[target].astype(int).to_numpy()
    fovs = frame["fov"].astype(str).to_numpy()
    classes = np.sort(np.unique(y))
    strata = joint_strata(frame, target).to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_raw = np.zeros((len(frame), len(classes)), dtype=float)
    oof_pred = np.zeros(len(frame), dtype=int)
    fold_metrics = []
    fold_thresholds = []

    for fold, (outer_train, outer_test) in enumerate(outer.split(frame, strata)):
        print(f"  {task}/{variant}: outer fold {fold + 1}/5", flush=True)
        train_frame = frame.iloc[outer_train].copy()
        train_y = y[outer_train]
        train_fov = fovs[outer_train]
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fold)
        inner_raw = np.zeros((len(outer_train), len(classes)), dtype=float)
        for inner_train, inner_valid in inner.split(train_frame, strata[outer_train]):
            fit_frame = train_frame.iloc[inner_train]
            valid_frame = train_frame.iloc[inner_valid]
            transform = fit_variant_transformer(fit_frame, variant)
            fit_x = transform_variant(fit_frame, transform)
            valid_x = transform_variant(valid_frame, transform)
            model = classifier().fit(fit_x, train_y[inner_train])
            inner_raw[inner_valid] = model.predict_proba(valid_x)

        calibrators = fit_fov_calibrators(inner_raw, train_y, train_fov, classes)
        inner_calibrated = apply_fov_calibrators(inner_raw, train_fov, classes, calibrators)
        thresholds = {
            fov: dr_threshold(train_y[train_fov == fov], inner_calibrated[train_fov == fov], classes, task)
            for fov in ("3mm", "6mm")
        }

        transform = fit_variant_transformer(train_frame, variant)
        fit_x = transform_variant(train_frame, transform)
        test_x = transform_variant(frame.iloc[outer_test], transform)
        model = classifier().fit(fit_x, train_y)
        test_raw = model.predict_proba(test_x)
        oof_raw[outer_test] = test_raw
        test_calibrated = apply_fov_calibrators(test_raw, fovs[outer_test], classes, calibrators)
        for row_index, row in enumerate(outer_test):
            oof_pred[row] = predict_at_threshold(
                test_calibrated[row_index:row_index + 1], classes,
                thresholds[fovs[row]], task,
            )[0]
        fold_metrics.append(metrics(y[outer_test], oof_pred[outer_test]))
        fold_thresholds.append(thresholds)

    pooled = metrics(y, oof_pred)
    matrix = confusion_matrix(y, oof_pred, labels=classes)
    final_calibrators = fit_fov_calibrators(oof_raw, y, fovs, classes)
    calibrated_oof = apply_fov_calibrators(oof_raw, fovs, classes, final_calibrators)
    thresholds = {
        fov: dr_threshold(y[fovs == fov], calibrated_oof[fovs == fov], classes, task)
        for fov in ("3mm", "6mm")
    }
    fitted_transform = fit_variant_transformer(frame, variant)
    full_x = transform_variant(frame, fitted_transform)
    fitted_classifier = classifier().fit(full_x, y)
    return {
        "model": "ExtraTreesClassifier",
        "metrics": pooled,
        "fold_metrics": fold_metrics,
        "fold_fov_dr_thresholds": fold_thresholds,
        "thresholds_by_fov": thresholds,
        "confusion_matrix_oof": matrix.tolist(),
        "class_names": [CLASS_NAMES[task][int(i)] for i in classes],
        "pca_components": 32 if variant in ("cnn_baseline", "hybrid") else None,
        "pca_fit": "training rows only within each inner/outer CV fold",
        "fov_normalization": "z-score fitted separately per FOV within each training fold",
        "probability_calibration": "per-FOV one-vs-rest sigmoid fitted on out-of-fold probabilities",
    }, {
        "variant": variant,
        "transformer": fitted_transform,
        "classifier": fitted_classifier,
        "classes": [int(i) for i in fitted_classifier.classes_],
        "class_names": CLASS_NAMES[task],
        "probability_calibrators": final_calibrators,
        "thresholds_by_fov": thresholds,
        "transformed_feature_names": list(full_x.columns),
    }


def train():
    features_path = OUTPUTS / "topology_features_500.csv"
    if not features_path.is_file():
        raise FileNotFoundError("Run scripts/extract_features.py first to create CNN embeddings")
    data = pd.read_csv(features_path, dtype={"id": str})
    expected_cnn = [f"cnn_{index:03d}" for index in range(EMBEDDING_DIM)]
    missing = [name for name in expected_cnn if name not in data.columns]
    if missing:
        raise ValueError(f"Feature table is missing {len(missing)} ResNet18 embeddings")

    summary_path = OUTPUTS / "feature_extraction_summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    results.update({
        "experiment": "Frozen ResNet18 CNN + vascular topology ablation",
        "cnn_embedding_dim": EMBEDDING_DIM,
        "cross_validation": "Nested StratifiedKFold(5 outer, 4 inner), stratified by FOV and class",
        "class_weight": "balanced",
        "random_state": 42,
        "excluded_demographics": ["age", "sex"],
        "cnn": CNN_CONFIG,
        "pca_components": 32,
        "tasks": {},
    })
    model_artifact = {
        "cnn_config": CNN_CONFIG,
        "cnn_state_dict": torch.load(
            OUTPUTS / "resnet18_imagenet1k_v1.pth", map_location="cpu", weights_only=True
        ),
        "tasks": {},
        "model_names": {
            "cnn_baseline": "CNN Baseline",
            "topology_only": "Topology Only",
            "hybrid": "Hybrid CNN + Topology",
        },
    }

    for task, target in (("binary", "binary_label"), ("multiclass", "multiclass_label")):
        subset = data.dropna(subset=[target]).copy().reset_index(drop=True)
        results["tasks"][task] = {"class_names": CLASS_NAMES[task], "models": {}}
        model_artifact["tasks"][task] = {"models": {}}
        for variant in MODEL_VARIANTS:
            outcome, fitted = evaluate_variant(subset, target, task, variant)
            results["tasks"][task]["models"][variant] = outcome
            model_artifact["tasks"][task]["models"][variant] = fitted

            display = ConfusionMatrixDisplay(
                np.asarray(outcome["confusion_matrix_oof"]),
                display_labels=outcome["class_names"],
            )
            display.plot(cmap="Blues")
            display.ax_.set_title(f"{variant.replace('_', ' ').title()} ({task}) — nested OOF")
            plt.tight_layout()
            plt.savefig(OUTPUTS / f"cm_{task}_{variant}.png", dpi=150)
            if task == "binary" and variant in ("cnn_baseline", "hybrid"):
                alias = "cm_baseline.png" if variant == "cnn_baseline" else "cm_baseline_topology.png"
                plt.savefig(OUTPUTS / alias, dpi=150)
            plt.close()

            if task == "multiclass" and variant == "hybrid":
                fitted_model = fitted["classifier"]
                importances = fitted_model.feature_importances_
                order = np.argsort(importances)[-20:]
                plt.figure(figsize=(9, 6))
                plt.barh([fitted["transformed_feature_names"][i] for i in order], importances[order])
                plt.title("Hybrid CNN + topology feature importance")
                plt.tight_layout()
                plt.savefig(OUTPUTS / "feature_importance.png", dpi=150)
                plt.close()

    with (OUTPUTS / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (OUTPUTS / "dr_classifier.pkl").open("wb") as handle:
        pickle.dump(model_artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    train()
