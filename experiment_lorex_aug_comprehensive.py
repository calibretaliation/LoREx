#!/usr/bin/env python
# coding: utf-8

# # Comprehensive LoREx-Aug Evaluation
# 
# **Method:** LoREx-Aug — whiten a test feature using pre-computed trusted statistics, score by absolute projection onto the top eigenvector of the whitened trusted covariance.
# 
# **Experiments:**
# 1. **Main results** — 5 attacks with default settings
# 2. **Poison rate sweep** — vary N_POISON ∈ {1, 2, 5, 10, 20, 50, 100, 200} out of 1000
# 3. **Trusted set size sweep** — vary N_TRUSTED ∈ {100, 200, 500, 1000, 2000, 5000}
# 4. **Trusted dataset comparison** — different clean reference sets (STL-10, SVHN, GTSRB, ImageNet, CIFAR-10)
# 
# **Attacks tested:**
# 
# | Attack | Encoder | Input | Trigger |
# |--------|---------|-------|---------|
# | DRUPE-CLIP | CLIP-RN50 | 224×224 | NPZ patch |
# | BadCLIP | CLIP-RN50 | 224×224 | Optimized patch |
# | INACTIVE-CLIP | CLIP-RN50 | 224×224 | UNet filter |
# | BadEncoder-SimCLR | SimCLR (ResNet) | 32×32 | NPZ patch |
# | DRUPE-SimCLR | SimCLR (ResNet) | 32×32 | NPZ patch |

# In[1]:


"""Cell 1: Imports & Path Setup."""

import sys
import os
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets as tv_datasets
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# get_ipython().run_line_magic('matplotlib', 'inline')

# ── Project paths ──
REPO_ROOT = Path("/media/lambda/SSD1/nhat/LoREx")
for p in [str(REPO_ROOT), str(REPO_ROOT / "attacks"), str(REPO_ROOT / "defenses")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── LoREx imports ──
from attacks import (
    build_drupe_clip_attack,
    build_badclip_attack,
    build_badencoder_attack,
    build_drupe_attack,
    build_inactive_attack,
    AttackSpec,
)
from dataset import build_trusted_loader, NpzCleanDataset
from lorex.whitening import compute_whitening_matrix
from lorex.metrics import full_metrics, compute_auc
from lorex.features import extract_features

print("All imports OK")


# In[2]:


"""Cell 2: Master Configuration."""

# ═══════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════
DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4

device = torch.device(DEVICE)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════════════
# DATA ROOTS
# ═══════════════════════════════════════════════════════════════════
DATA_ROOT = "/media/lambda/SSD1/nhat/data"
IMAGENET_TRAIN = f"{DATA_ROOT}/imagenet/train"

# Sources for the realistic mixed CLIP test dataset
IMAGENET_TRAIN_DIR = IMAGENET_TRAIN
CIFAR10_NPZ = f"{DATA_ROOT}/cifar10/train.npz"
STL10_NPZ = f"{DATA_ROOT}/stl10/train.npz"

# ═══════════════════════════════════════════════════════════════════
# ATTACK CONFIGS
# ═══════════════════════════════════════════════════════════════════
ATTACK_CONFIGS = [
    {
        "name": "DRUPE-CLIP",
        "builder": "build_drupe_clip_attack",
        "builder_kwargs": {
            "ckpt_path": str(REPO_ROOT / "attacks/DRUPE/output/CLIP/backdoor/truck/model_200.pth"),
        },
        "model_family": "clip",
        "trigger_type": "npz",
        "trigger_path": str(REPO_ROOT / "attacks/DRUPE/trigger/trigger_pt_white_185_24.npz"),
        "trigger_size": 224,
    },
    {
        "name": "BadCLIP",
        "builder": "build_badclip_attack",
        "builder_kwargs": {
            "ckpt_path": str(REPO_ROOT / "attacks/BadCLIP/badclip/logs/nodefence_ours_final/checkpoints/epoch_10.pt"),
        },
        "model_family": "clip",
        "trigger_type": "badclip",
        "patch_type": "ours_tnature",
        "patch_location": "middle",
        "patch_size": 16,
        "patch_name": str(REPO_ROOT / "attacks/BadCLIP/opti_patches/tnature_eda_aug_bs64_ep50_16_middle_01_05_pos_neg_tri*500.jpg"),
    },
    {
        "name": "INACTIVE-CLIP",
        "builder": "build_inactive_attack",
        "builder_kwargs": {
            "model_type": "clip",
            "clip_ckpt_path": str(REPO_ROOT / "attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth"),
            "unet_path": str(REPO_ROOT / "attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt"),
            "config_path": None,
            "encoder_path": None,
        },
        "model_family": "clip",
        "trigger_type": "unet",
    },
    {
        "name": "BadEncoder-SimCLR",
        "builder": "build_badencoder_attack",
        "builder_kwargs": {
            "ckpt_path": str(REPO_ROOT / "attacks/BadEncoder/output/cifar10/gtsrb_backdoored_encoder/model_200.pth"),
            "usage_info": "cifar10",
        },
        "model_family": "simclr",
        "trigger_type": "npz",
        "trigger_path": str(REPO_ROOT / "attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz"),
        "trigger_size": 32,
    },
    {
        "name": "DRUPE-SimCLR",
        "builder": "build_drupe_attack",
        "builder_kwargs": {
            "ckpt_path": str(REPO_ROOT / "attacks/DRUPE/output/cifar10/gtsrb_backdoored_encoder/model_200.pth"),
        },
        "model_family": "simclr",
        "trigger_type": "npz",
        "trigger_path": str(REPO_ROOT / "attacks/DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz"),
        "trigger_size": 32,
    },
]

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT DEFAULTS
# ═══════════════════════════════════════════════════════════════════
DEFAULT_N_TOTAL = 1000
DEFAULT_N_POISON = 10
DEFAULT_N_TRUSTED = 2000
DEFAULT_TRUSTED_CLIP = "stl10"
DEFAULT_TRUSTED_SIMCLR = "cifar10"

# ═══════════════════════════════════════════════════════════════════
# SWEEP PARAMETERS
# ═══════════════════════════════════════════════════════════════════
POISON_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200]
TRUSTED_SIZES = [100, 200, 500, 1000, 2000, 5000]

TRUSTED_DATASETS_CLIP = ["stl10", "svhn", "gtsrb", "imagenet"]
TRUSTED_DATASETS_SIMCLR = ["cifar10", "stl10", "svhn", "gtsrb"]

# ═══════════════════════════════════════════════════════════════════
# CLIP MIXED DATASET COMPOSITION
# ═══════════════════════════════════════════════════════════════════
N_FROM_IMAGENET = 500
N_FROM_CIFAR10 = 300
N_FROM_STL10 = 200

print(f"Device: {device}")
print(f"Attacks: {[c['name'] for c in ATTACK_CONFIGS]}")
print(f"Poison counts: {POISON_COUNTS}")
print(f"Trusted sizes: {TRUSTED_SIZES}")


# In[3]:


"""Cell 3: Attack Loading / Unloading Helpers."""

BUILDER_MAP = {
    "build_drupe_clip_attack": build_drupe_clip_attack,
    "build_badclip_attack": build_badclip_attack,
    "build_inactive_attack": build_inactive_attack,
    "build_badencoder_attack": build_badencoder_attack,
    "build_drupe_attack": build_drupe_attack,
}


def load_attack(cfg, device):
    """Load an attack from a config dict. Returns AttackSpec."""
    fn = BUILDER_MAP[cfg["builder"]]
    kwargs = {**cfg["builder_kwargs"], "device": device}
    attack = fn(**kwargs)
    print(f"  Loaded {cfg['name']} ({cfg['model_family']}) on {device}")
    return attack


def unload_attack(attack):
    """Move model off GPU and free memory."""
    attack.model.cpu()
    if attack.trigger_model is not None:
        attack.trigger_model.cpu()
    del attack
    gc.collect()
    torch.cuda.empty_cache()
    print("  Unloaded attack, GPU memory freed.")


# In[4]:


"""Cell 4: Test Dataset Builders.

Unified interface for all attack types. Returns (DataLoader, labels_array, needs_unet)
where labels are 0=clean, 1=poison, and needs_unet indicates UNet post-processing.

CLIP attacks: CLIPMixedTestDataset (500 ImageNet + 300 CIFAR10 + 200 STL10)
  - trigger_type "npz": apply mask/patch from NPZ in __getitem__
  - trigger_type "badclip": apply BadCLIP.backdoor.utils.apply_trigger() in __getitem__
  - trigger_type "unet": dataset returns CLEAN tensors; UNet applied during feature extraction

SimCLR attacks: SimCLRTestDataset (CIFAR-10 test.npz)
  - trigger_type "npz": apply mask/patch from NPZ in __getitem__

NOTE: No CUDA operations in any Dataset __getitem__ — all GPU work happens in
feature extraction. This avoids CUDA errors with DataLoader workers.
"""

# ── Data caches (avoid redundant I/O across sweep iterations) ──
_clip_pil_cache = {}      # (n_total, seed) -> list[PIL.Image]
_cifar10_test_cache = None # np.ndarray (10000, 32, 32, 3)


class CLIPMixedTestDataset(Dataset):
    """Mixed-source test dataset for CLIP attacks.

    Applies NPZ or BadCLIP triggers at PIL level in __getitem__.
    For UNet triggers: returns clean tensors (UNet applied later during extraction).
    No CUDA operations here — safe for multi-worker DataLoader.
    """

    def __init__(self, pil_images, poison_flags, trigger_cfg, transform):
        self.images = pil_images
        self.poison_flags = poison_flags
        self.trigger_cfg = trigger_cfg
        self.transform = transform

        # Pre-load trigger data for NPZ type
        if trigger_cfg["trigger_type"] == "npz":
            td = np.load(trigger_cfg["trigger_path"])
            self.trigger_patch = td["t"]
            self.trigger_mask = td["tm"]
            self.trigger_size = trigger_cfg["trigger_size"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        is_poison = int(self.poison_flags[idx])
        ttype = self.trigger_cfg["trigger_type"]

        if is_poison and ttype == "npz":
            img_np = np.array(img.resize(
                (self.trigger_size, self.trigger_size), Image.BILINEAR
            ))
            img_np = (img_np * self.trigger_mask + self.trigger_patch).astype(np.uint8)
            img = Image.fromarray(img_np)

        elif is_poison and ttype == "badclip":
            from BadCLIP.backdoor.utils import apply_trigger
            args = SimpleNamespace(
                patch_size=self.trigger_cfg["patch_size"],
                patch_name=self.trigger_cfg["patch_name"],
                scale=None,
            )
            img = apply_trigger(
                img,
                patch_size=self.trigger_cfg["patch_size"],
                patch_type=self.trigger_cfg["patch_type"],
                patch_location=self.trigger_cfg["patch_location"],
                args=args,
            )

        # For "unet" trigger_type: return clean tensor — UNet applied during extraction
        img_tensor = self.transform(img)
        return img_tensor, is_poison


class SimCLRTestDataset(Dataset):
    """Test dataset for SimCLR attacks with NPZ patch trigger.

    Returns (img_tensor, is_poison) — same interface as CLIPMixedTestDataset.
    No CUDA operations — safe for multi-worker DataLoader.
    """

    def __init__(self, images_np, poison_flags, trigger_path, trigger_size,
                 transform):
        self.images = images_np
        self.poison_flags = poison_flags
        self.transform = transform
        self.trigger_size = trigger_size

        td = np.load(trigger_path)
        self.trigger_patch = td["t"]
        self.trigger_mask = td["tm"]

        if self.trigger_patch.ndim == 4:
            self.trigger_patch = self.trigger_patch[0]
        if self.trigger_mask.ndim == 4:
            self.trigger_mask = self.trigger_mask[0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx].copy()
        is_poison = int(self.poison_flags[idx])

        if is_poison:
            import cv2
            ori_h, ori_w = img_np.shape[:2]
            if self.trigger_size and (ori_h != self.trigger_size or ori_w != self.trigger_size):
                img_np = cv2.resize(img_np, (self.trigger_size, self.trigger_size))
            img_np = (img_np * self.trigger_mask + self.trigger_patch).astype(np.uint8)
            if self.trigger_size and (ori_h != self.trigger_size or ori_w != self.trigger_size):
                img_np = cv2.resize(img_np, (ori_w, ori_h))

        img_tensor = self.transform(Image.fromarray(img_np))
        return img_tensor, is_poison


def _collect_clip_pil_images(n_total, seed):
    """Collect PIL images from ImageNet + CIFAR10 + STL10. Results are cached."""
    key = (n_total, seed)
    if key in _clip_pil_cache:
        print(f"  [cache hit] Reusing {n_total} CLIP PIL images (seed={seed})")
        return _clip_pil_cache[key]

    rng = np.random.default_rng(seed)
    all_pil = []

    # ImageNet
    imagenet_ds = tv_datasets.ImageFolder(IMAGENET_TRAIN_DIR)
    indices = rng.choice(len(imagenet_ds), size=N_FROM_IMAGENET, replace=False)
    for i in tqdm(indices, desc="ImageNet", leave=False):
        path, _ = imagenet_ds.samples[i]
        all_pil.append(Image.open(path).convert("RGB"))

    # CIFAR-10
    cifar_x = np.load(CIFAR10_NPZ)["x"]
    indices = rng.choice(len(cifar_x), size=N_FROM_CIFAR10, replace=False)
    for i in tqdm(indices, desc="CIFAR10", leave=False):
        all_pil.append(Image.fromarray(cifar_x[i]))

    # STL-10
    stl_x = np.load(STL10_NPZ)["x"]
    n_avail = len(stl_x)
    indices = rng.choice(n_avail, size=min(N_FROM_STL10, n_avail),
                         replace=N_FROM_STL10 > n_avail)
    for i in tqdm(indices, desc="STL10", leave=False):
        all_pil.append(Image.fromarray(stl_x[i]))

    assert len(all_pil) == n_total
    _clip_pil_cache[key] = all_pil
    return all_pil


def _get_cifar10_test():
    """Load CIFAR-10 test images. Cached after first call."""
    global _cifar10_test_cache
    if _cifar10_test_cache is None:
        _cifar10_test_cache = np.load(f"{DATA_ROOT}/cifar10/test.npz")["x"]
    return _cifar10_test_cache


def build_test_dataset(cfg, attack, n_total, n_poison, seed):
    """Build test dataset for any attack config.

    Returns: (DataLoader, labels_array, needs_unet)
      - needs_unet: True if UNet trigger must be applied during feature extraction
    """
    rng = np.random.default_rng(seed)
    needs_unet = (cfg["trigger_type"] == "unet")

    if cfg["model_family"] == "clip":
        pil_images = _collect_clip_pil_images(n_total, seed)

        poison_flags = np.zeros(n_total, dtype=bool)
        poison_flags[rng.choice(n_total, size=n_poison, replace=False)] = True

        dataset = CLIPMixedTestDataset(
            pil_images=pil_images,
            poison_flags=poison_flags,
            trigger_cfg=cfg,
            transform=attack.processor.process_image,
        )

    else:  # simclr
        cifar_x = _get_cifar10_test()

        indices = rng.choice(len(cifar_x), size=n_total, replace=False)
        images = cifar_x[indices]

        poison_flags = np.zeros(n_total, dtype=bool)
        poison_flags[rng.choice(n_total, size=n_poison, replace=False)] = True

        dataset = SimCLRTestDataset(
            images_np=images,
            poison_flags=poison_flags,
            trigger_path=cfg["trigger_path"],
            trigger_size=cfg["trigger_size"],
            transform=attack.transform,
        )

    labels = poison_flags.astype(int)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return loader, labels, needs_unet


print("Test dataset builders ready.")


# In[5]:


"""Cell 5: Core LoREx-Aug Pipeline Functions.

precompute_lorex_aug_stats    — compute mu, W, top_eigvec from trusted features (all CPU)
score_lorex_aug               — score features using pre-computed stats (all CPU)
extract_test_features         — extract L2-normalized features → CPU tensors
extract_test_features_unet    — same but with UNet trigger applied to poison images on GPU
run_lorex_aug                 — full pipeline with optional precomputed caching
"""


def precompute_lorex_aug_stats(trusted_feats):
    """Compute LoREx-Aug statistics from trusted features.

    All returned tensors are on CPU to avoid orphaned GPU memory when the
    attack model is later unloaded.

    Returns dict with keys: mu (CPU tensor), W (CPU tensor), top_eigvec (numpy).
    """
    mu = trusted_feats.mean(dim=0).cpu()                              # (d,) CPU
    W = compute_whitening_matrix(
        trusted_feats - trusted_feats.mean(dim=0), method="cholesky", ridge=1e-3
    ).cpu()                                                            # (d, d) CPU
    tw = ((trusted_feats - trusted_feats.mean(dim=0)).cpu() @ W).numpy()  # (n, d)
    cov_w = np.cov(tw.T)                                              # (d, d)
    ev, evec = np.linalg.eigh(cov_w)
    idx = np.argsort(ev)[::-1]
    return {"mu": mu, "W": W, "top_eigvec": evec[:, idx[0]]}


def score_lorex_aug(feats, stats):
    """Score features using pre-computed LoREx-Aug stats. All on CPU.

    Args:
        feats: (N, d) CPU tensor
        stats: dict from precompute_lorex_aug_stats (CPU tensors)

    Returns: (N,) numpy scores — higher → more likely poisoned.
    """
    z = ((feats - stats["mu"]) @ stats["W"]).numpy()                  # (N, d)
    return np.abs(z @ stats["top_eigvec"])                             # (N,)


def extract_test_features(loader, model, device, feature_fn):
    """Extract L2-normalized features and labels from test loader.

    Features are moved to CPU immediately to avoid GPU memory accumulation.
    Returns: (feats_cpu_tensor, labels_numpy)
    """
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Feature extraction", leave=False):
            imgs = imgs.to(device)
            feats = F.normalize(feature_fn(model, imgs), dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels.numpy())
    return torch.cat(all_feats, 0), np.concatenate(all_labels)


def extract_test_features_unet(loader, model, trigger_model, device, feature_fn):
    """Extract features with UNet trigger applied to poisoned images.

    For clean images: extract features directly.
    For poison-flagged images: apply UNet trigger on GPU, then extract features.
    All done in batches on GPU for efficiency, results moved to CPU.

    Returns: (feats_cpu_tensor, labels_numpy)
    """
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="UNet feature extraction", leave=False):
            imgs = imgs.to(device)
            labels_np = labels.numpy()

            # Start with clean features for all images
            batch_feats = F.normalize(feature_fn(model, imgs), dim=-1)

            # For poisoned images, re-extract with UNet trigger
            poison_mask = labels_np == 1
            if poison_mask.any():
                poison_idx = torch.tensor(poison_mask).to(device)
                poisoned_imgs = trigger_model(imgs[poison_idx])
                poisoned_imgs = torch.clamp(poisoned_imgs, imgs.min(), imgs.max())
                poison_feats = F.normalize(feature_fn(model, poisoned_imgs), dim=-1)
                batch_feats[poison_idx] = poison_feats

            all_feats.append(batch_feats.cpu())
            all_labels.append(labels_np)
    return torch.cat(all_feats, 0), np.concatenate(all_labels)


def run_lorex_aug(
    cfg, attack, trusted_dataset, n_trusted, n_total, n_poison,
    device, seed=42,
    precomputed_stats=None,
    precomputed_test=None,
):
    """Run the full LoREx-Aug pipeline for one configuration.

    Args:
        cfg: attack config dict
        attack: loaded AttackSpec
        trusted_dataset: str (e.g. "stl10", "cifar10", "imagenet")
        n_trusted: number of trusted samples
        n_total: total test images
        n_poison: number of poisoned test images
        device: torch device
        seed: random seed
        precomputed_stats: if provided, skip trusted feature extraction
        precomputed_test: if provided (feats, labels), skip test dataset build

    Returns: dict with metrics and scores.
    """
    t0 = time.time()

    # ── Step 1: Trusted features + stats ──
    if precomputed_stats is not None:
        stats = precomputed_stats
    else:
        use_224 = (cfg["model_family"] == "clip")
        transform = attack.processor.process_image if cfg["model_family"] == "clip" else attack.transform
        if trusted_dataset == "imagenet":
            data_dir = IMAGENET_TRAIN
        else:
            data_dir = f"{DATA_ROOT}/{trusted_dataset}"

        trusted_loader = build_trusted_loader(
            trusted_dataset=trusted_dataset,
            data_dir=data_dir,
            transform=transform,
            n_trusted=n_trusted,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_224=use_224,
            split="train",
            seed=seed,
        )
        trusted_feats = extract_features(
            trusted_loader, attack.model, device, attack.feature_fn
        )
        stats = precompute_lorex_aug_stats(trusted_feats)

    # ── Step 2: Test features + labels ──
    if precomputed_test is not None:
        all_feats, all_labels = precomputed_test
    else:
        test_loader, labels, needs_unet = build_test_dataset(
            cfg, attack, n_total, n_poison, seed
        )
        if needs_unet:
            all_feats, all_labels = extract_test_features_unet(
                test_loader, attack.model, attack.trigger_model,
                device, attack.feature_fn
            )
        else:
            all_feats, all_labels = extract_test_features(
                test_loader, attack.model, device, attack.feature_fn
            )

    # ── Step 3: Score (all on CPU) ──
    scores = score_lorex_aug(all_feats, stats)

    clean_scores = scores[all_labels == 0]
    poison_scores = scores[all_labels == 1]

    # ── Step 4: Metrics ──
    m = full_metrics(clean_scores, poison_scores)
    elapsed = time.time() - t0

    return {
        "attack": cfg["name"],
        "trusted_dataset": trusted_dataset,
        "n_trusted": n_trusted,
        "n_total": n_total,
        "n_poison": n_poison,
        "poison_rate": n_poison / n_total,
        "auc": m["auc"],
        "tpr@1%": m["tpr@1%"],
        "tpr@5%": m["tpr@5%"],
        "tpr@10%": m["tpr@10%"],
        "clean_scores": clean_scores,
        "poison_scores": poison_scores,
        "fpr": m["fpr"],
        "tpr": m["tpr"],
        "elapsed_s": elapsed,
    }


def results_to_df(results_list):
    """Convert list of result dicts to a clean DataFrame (no array columns)."""
    return pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("clean_scores", "poison_scores", "fpr", "tpr")}
        for r in results_list
    ])


print("LoREx-Aug pipeline functions ready.")


# ## Experiment 1: Main Results Across All Attacks
# 
# Run LoREx-Aug on all 5 attacks with default settings (N_TOTAL=1000, N_POISON=10, N_TRUSTED=2000).

# In[6]:


"""Experiment 1: Main Results — All 5 Attacks with Default Settings."""

