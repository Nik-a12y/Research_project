"""Fold-fitted FOV normalization and CNN PCA transforms for the ablation models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cnn_embeddings import EMBEDDING_DIM
from src.features import TOPO_NAMES


CNN_NAMES = [f"cnn_{index:03d}" for index in range(EMBEDDING_DIM)]
MODEL_VARIANTS = ("cnn_baseline", "topology_only", "hybrid")
PCA_COMPONENTS = 32


def _fit_fov_stats(frame: pd.DataFrame, names: list[str]) -> dict:
    stats = {}
    for fov in ("3mm", "6mm"):
        values = frame.loc[frame["fov"] == fov, names].to_numpy(dtype=np.float64)
        if not len(values):
            raise ValueError(f"Cannot fit FOV normalization without {fov} rows")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        stats[fov] = {"mean": mean, "scale": scale}
    return stats


def _apply_fov_stats(frame: pd.DataFrame, names: list[str], stats: dict) -> np.ndarray:
    values = frame[names].to_numpy(dtype=np.float64).copy()
    fovs = frame["fov"].astype(str).to_numpy()
    for fov in ("3mm", "6mm"):
        mask = fovs == fov
        if mask.any():
            values[mask] = (values[mask] - stats[fov]["mean"]) / stats[fov]["scale"]
    return values


def fit_variant_transformer(frame: pd.DataFrame, variant: str) -> dict:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    transformer = {"variant": variant}
    if variant in ("cnn_baseline", "hybrid"):
        transformer["cnn_fov_stats"] = _fit_fov_stats(frame, CNN_NAMES)
        normalized = _apply_fov_stats(frame, CNN_NAMES, transformer["cnn_fov_stats"])
        transformer["cnn_scaler"] = StandardScaler().fit(normalized)
        scaled = transformer["cnn_scaler"].transform(normalized)
        transformer["pca"] = PCA(
            n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=42
        ).fit(scaled)
    if variant in ("topology_only", "hybrid"):
        transformer["topology_fov_stats"] = _fit_fov_stats(frame, list(TOPO_NAMES))
    return transformer


def transform_variant(frame: pd.DataFrame, transformer: dict) -> pd.DataFrame:
    variant = transformer["variant"]
    columns = {}
    if variant in ("cnn_baseline", "hybrid"):
        normalized = _apply_fov_stats(frame, CNN_NAMES, transformer["cnn_fov_stats"])
        scaled = transformer["cnn_scaler"].transform(normalized)
        reduced = transformer["pca"].transform(scaled)
        columns.update({f"cnn_pc_{i:02d}": reduced[:, i] for i in range(PCA_COMPONENTS)})
    if variant in ("topology_only", "hybrid"):
        normalized_topology = _apply_fov_stats(
            frame, list(TOPO_NAMES), transformer["topology_fov_stats"]
        )
        columns.update({name: normalized_topology[:, i] for i, name in enumerate(TOPO_NAMES)})
    return pd.DataFrame(columns, index=frame.index)
