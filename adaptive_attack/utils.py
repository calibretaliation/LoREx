import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_encoder_submodule(model: nn.Module, encoder_usage_info: str) -> nn.Module:
    if encoder_usage_info in ["cifar10", "stl10"]:
        return model.f
    if encoder_usage_info in ["imagenet", "CLIP"]:
        return model.visual
    raise ValueError(f"Unknown encoder_usage_info: {encoder_usage_info}")


def encode_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feature = encoder(x)
    return F.normalize(feature, dim=-1)


@torch.no_grad()
def compute_target_embedding(encoder: nn.Module, target_loader, device: torch.device) -> torch.Tensor:
    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(device, non_blocking=True)
            emb = encode_features(encoder, images)
            embeddings.append(emb)
    if not embeddings:
        raise RuntimeError("Target loader is empty; unable to compute target embedding")
    embedding = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
    return F.normalize(embedding, dim=-1)


@torch.no_grad()
def compute_low_variance_subspace(encoder: nn.Module, data_loader, device: torch.device, eig_k: int, max_batches: Optional[int] = None) -> torch.Tensor:
    encoder.eval()
    features = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            img_clean = batch[0].to(device, non_blocking=True)
            feat = encode_features(encoder, img_clean)
            features.append(feat.cpu())
            if max_batches is not None and (idx + 1) >= max_batches:
                break

    if not features:
        raise RuntimeError("Failed to collect features for eigen decomposition")

    feats = torch.cat(features, dim=0)
    feats = feats - feats.mean(dim=0, keepdim=True)
    cov = torch.matmul(feats.T, feats) / max(feats.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    sorted_idx = torch.argsort(eigvals)
    eig_k = min(eig_k, eigvecs.shape[1])
    u_low = eigvecs[:, sorted_idx[:eig_k]]
    return u_low.to(device)
