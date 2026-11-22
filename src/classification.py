"""FOV-domain normalization and probability calibration helpers."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_fov_standardizer(frame: pd.DataFrame, feature_names: list[str]) -> dict:
    """Fit per-FOV z-score statistics from the supplied training rows."""
    numeric = [name for name in feature_names if name != "fov_mm"]
    statistics = {}
    for fov in ("3mm", "6mm"):
        rows = frame.loc[frame["fov"] == fov, numeric]
        if rows.empty:
            raise ValueError(f"Training fold has no rows for FOV {fov}")
        values = rows.to_numpy(dtype=float)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        statistics[fov] = {"mean": mean, "scale": scale}
    return statistics


def apply_fov_standardizer(
    frame: pd.DataFrame,
    feature_names: list[str],
    statistics: dict,
) -> pd.DataFrame:
    """Apply saved domain statistics and append the raw numeric FOV feature."""
    numeric = [name for name in feature_names if name != "fov_mm"]
    values = frame[numeric].to_numpy(dtype=float).copy()
    fovs = frame["fov"].astype(str).to_numpy()
    for fov in ("3mm", "6mm"):
        mask = fovs == fov
        if mask.any():
            stats = statistics[fov]
            values[mask] = (values[mask] - stats["mean"]) / stats["scale"]
    normalized = pd.DataFrame(values, columns=numeric, index=frame.index)
    if "fov_mm" in feature_names:
        normalized["fov_mm"] = frame["fov"].str.removesuffix("mm").astype(float)
    return normalized[feature_names]


def fit_fov_calibrators(probabilities, labels, fovs, classes) -> dict:
    """Fit one-vs-rest sigmoid calibrators per FOV from out-of-fold scores."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fovs = np.asarray(fovs, dtype=str)
    calibrators = {}
    for fov in ("3mm", "6mm"):
        rows = fovs == fov
        calibrators[fov] = {}
        for column, class_id in enumerate(classes):
            target = (labels[rows] == class_id).astype(int)
            if np.unique(target).size < 2:
                raise ValueError(f"Cannot calibrate class {class_id} for {fov}: one label only")
            calibrator = LogisticRegression(C=1.0, random_state=42)
            calibrator.fit(probabilities[rows, column].reshape(-1, 1), target)
            calibrators[fov][int(class_id)] = calibrator
    return calibrators


def apply_fov_calibrators(probabilities, fovs, classes, calibrators) -> np.ndarray:
    """Calibrate class scores by FOV and normalize them to sum to one."""
    probabilities = np.asarray(probabilities, dtype=float)
    fovs = np.asarray(fovs, dtype=str)
    calibrated = np.zeros_like(probabilities, dtype=float)
    for fov in ("3mm", "6mm"):
        rows = fovs == fov
        if not rows.any():
            continue
        for column, class_id in enumerate(classes):
            calibrator = calibrators[fov][int(class_id)]
            calibrated[rows, column] = calibrator.predict_proba(
                probabilities[rows, column].reshape(-1, 1)
            )[:, 1]
    totals = calibrated.sum(axis=1, keepdims=True)
    return np.divide(calibrated, totals, out=np.full_like(calibrated, 1 / len(classes)), where=totals > 0)


def dr_threshold(labels, probabilities, classes, task: str) -> float:
    """Choose the DR threshold that maximizes F2 on calibration rows."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    dr_column = list(classes).index(1)
    from sklearn.metrics import fbeta_score

    candidates = np.linspace(0.02, 0.98, 97)
    best_score, best_threshold = -1.0, 0.5
    for threshold in candidates:
        labels_by_class = np.asarray(classes, dtype=int)
        dr_rows = probabilities[:, dr_column] >= threshold
        if task == "binary":
            predicted = np.where(dr_rows, 1, 0)
        else:
            non_dr_columns = [i for i, label in enumerate(labels_by_class) if label != 1]
            non_dr = labels_by_class[non_dr_columns][
                np.argmax(probabilities[:, non_dr_columns], axis=1)
            ]
            predicted = np.where(dr_rows, 1, non_dr)
        score = fbeta_score(labels == 1, predicted == 1, beta=2, zero_division=0)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score, best_threshold = score, float(threshold)
    return best_threshold
