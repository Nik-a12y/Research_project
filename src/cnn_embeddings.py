"""Frozen ImageNet ResNet18 embeddings with one preprocessing path for batch and API."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


WEIGHT_ENUM = ResNet18_Weights.IMAGENET1K_V1
EMBEDDING_DIM = 512


class FrozenResNet18:
    """ResNet18 average-pool embeddings using the official ImageNet transforms."""

    def __init__(self, state_dict=None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = None if state_dict is not None else WEIGHT_ENUM
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device).eval()
        self.preprocess = WEIGHT_ENUM.transforms()

    def state_dict(self):
        return {name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()}

    @torch.inference_mode()
    def embed(self, images: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 2:
            images = images[None, ...]
        if images.ndim != 3:
            raise ValueError("Expected grayscale images shaped (N, H, W)")
        if batch_size is None:
            batch_size = 32 if self.device.type == "cuda" else 8

        outputs = []
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).unsqueeze(1)
            batch = batch.repeat(1, 3, 1, 1)
            batch = self.preprocess(batch).to(self.device)
            embedding = self.model(batch)
            if embedding.ndim != 2 or embedding.shape[1] != EMBEDDING_DIM:
                raise RuntimeError(f"Expected 512-D ResNet18 embeddings, got {tuple(embedding.shape)}")
            outputs.append(embedding.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, EMBEDDING_DIM), np.float32)
