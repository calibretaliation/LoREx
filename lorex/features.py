"""Feature extraction utilities for LoREx.

Provides functions to extract L2-normalized embeddings from SSL encoders for
trusted-clean and test (clean + poisoned) sets.
"""

from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


@torch.no_grad()
def extract_features(
    loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """Extract L2-normalized features from a DataLoader.

    Args:
        loader: DataLoader yielding batches of images (or (img, label) tuples).
        model: SSL encoder.
        device: target device.
        feature_fn: callable(model, img) -> raw embedding Tensor.
        max_batches: if set, stop after this many batches.

    Returns:
        Tensor of shape (n, d), L2-normalized.
    """
    feats = []
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        img = batch[0] if isinstance(batch, (list, tuple)) else batch
        img = _to_device(img, device)
        f = feature_fn(model, img)
        feats.append(F.normalize(f, dim=-1))
    return torch.cat(feats, dim=0)


@torch.no_grad()
def extract_clean_and_poison_features_pair(
    loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract clean and poisoned features from a pair-loader.

    Pair loaders yield (trigger_img, clean_img, label, trigger_label) tuples,
    providing pre-paired clean/poisoned images.

    Args:
        loader: pair DataLoader.
        model: SSL encoder.
        device: target device.
        feature_fn: callable(model, img) -> embedding.
        max_samples: stop after this many samples (approximate).

    Returns:
        (clean_feats, poison_feats), both L2-normalized, shape (n, d).
    """
    clean_feats, poison_feats = [], []
    n = 0
    for batch in loader:
        trigger_img, clean_img, _label, _label_trigger = batch
        clean_img = _to_device(clean_img, device)
        trigger_img = _to_device(trigger_img, device)

        fc = F.normalize(feature_fn(model, clean_img), dim=-1)
        fp = F.normalize(feature_fn(model, trigger_img), dim=-1)
        clean_feats.append(fc)
        poison_feats.append(fp)
        n += fc.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    return torch.cat(clean_feats, 0), torch.cat(poison_feats, 0)


@torch.no_grad()
def extract_clean_and_poison_features_trigger(
    loader: Iterable,
    model: torch.nn.Module,
    trigger_model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract clean and poisoned features using a trigger generator model.

    Used for attacks (e.g. INACTIVE) where triggers are generated on-the-fly
    by a learned model (e.g. UNet) rather than pre-applied to the dataset.

    Args:
        loader: DataLoader yielding (img, label) batches.
        model: SSL encoder.
        trigger_model: callable(img) -> poisoned_img.
        device: target device.
        feature_fn: callable(model, img) -> embedding.
        max_samples: stop after this many samples (approximate).

    Returns:
        (clean_feats, poison_feats), both L2-normalized, shape (n, d).
    """
    clean_feats, poison_feats = [], []
    n = 0
    for batch in loader:
        img = batch[0] if isinstance(batch, (list, tuple)) else batch
        img = _to_device(img, device)

        fc = F.normalize(feature_fn(model, img), dim=-1)
        poisoned_img = trigger_model(img)
        poisoned_img = torch.clamp(poisoned_img, img.min(), img.max())
        fp = F.normalize(feature_fn(model, poisoned_img), dim=-1)
        clean_feats.append(fc)
        poison_feats.append(fp)
        n += fc.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    return torch.cat(clean_feats, 0), torch.cat(poison_feats, 0)
