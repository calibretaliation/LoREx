import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
    precision_recall_curve,
    average_precision_score,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def cov_centered(x: torch.Tensor) -> torch.Tensor:
    x64 = x.double()
    x64 = x64 - x64.mean(dim=0, keepdim=True)
    n = x64.shape[0]
    denom = max(1, n - 1)
    return (x64.T @ x64) / denom


def compute_whitening_matrix(
    centered_clean: torch.Tensor,
    method: str = "cholesky",
    ridge: float = 1e-3,
    max_tries: int = 8,
) -> torch.Tensor:
    cov = cov_centered(centered_clean)
    d = cov.shape[0]
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    tr = torch.trace(cov)
    base = (tr / d).clamp_min(1e-12)
    lam0 = float(ridge) * float(base.item())

    if method == "eig":
        cov_reg = cov + lam0 * eye
        e_vals, e_vecs = torch.linalg.eigh(cov_reg)
        inv_sqrt = torch.rsqrt(torch.clamp(e_vals, min=torch.as_tensor(lam0, device=e_vals.device, dtype=e_vals.dtype)))
        w = e_vecs @ torch.diag(inv_sqrt) @ e_vecs.T
        return w.to(centered_clean.dtype)

    for i in range(int(max_tries)):
        lam = lam0 * (10.0 ** i)
        try:
            cov_reg = cov + lam * eye
            l = torch.linalg.cholesky(cov_reg)
            inv_l = torch.linalg.solve_triangular(l, eye, upper=False)
            w = inv_l.T
            return w.to(centered_clean.dtype)
        except RuntimeError:
            continue

    cov_reg = cov + lam0 * (10.0 ** (int(max_tries) - 1)) * eye
    e_vals, e_vecs = torch.linalg.eigh(cov_reg)
    inv_sqrt = torch.rsqrt(torch.clamp(e_vals, min=torch.as_tensor(lam0, device=e_vals.device, dtype=e_vals.dtype)))
    w = e_vecs @ torch.diag(inv_sqrt) @ e_vecs.T
    return w.to(centered_clean.dtype)


@torch.no_grad()
def extract_features(
    loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    feats = []
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            img = batch[0]
        else:
            img = batch
        img = to_device(img, device)
        f = feature_fn(model, img)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def extract_mixed_features_from_pair_loader(
    loader: Iterable,
    model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    trigger_chance: float = 0.01,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        trigger_img, img, label, _label_trigger = batch
        img = to_device(img, device)
        trigger_img = to_device(trigger_img, device)
        label = torch.as_tensor(label).long().to(device).view(-1)

        f = feature_fn(model, img)
        f = F.normalize(f, dim=-1)
        features.append(f)
        labels.append(torch.zeros_like(label))

        if np.random.rand() <= trigger_chance:
            f_trig = feature_fn(model, trigger_img)
            f_trig = F.normalize(f_trig, dim=-1)
            features.append(f_trig)
            labels.append(torch.ones_like(label))

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def extract_mixed_features_with_trigger_model(
    loader: Iterable,
    model: torch.nn.Module,
    trigger_model: torch.nn.Module,
    device: torch.device,
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    trigger_batch_num: Optional[int] = None,
    max_batches: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    features = []
    labels = []

    total_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    if total_batches <= 0:
        raise ValueError("total_batches must be >= 1")
    if trigger_batch_num is None:
        trigger_batch_num = total_batches // 100 + 1
    trigger_batch_num = int(min(max(1, trigger_batch_num), total_batches))
    rng = np.random.default_rng(int(seed))
    trigger_batch_ids = set(rng.choice(total_batches, size=trigger_batch_num, replace=False).tolist())

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= total_batches:
            break
        img, label = batch[0], batch[1]
        img = to_device(img, device)
        label = torch.as_tensor(label).long().to(device).view(-1)

        f_clean = feature_fn(model, img)
        f_clean = F.normalize(f_clean, dim=-1)
        features.append(f_clean)
        labels.append(torch.zeros_like(label))

        if b_idx in trigger_batch_ids:
            trig_img = trigger_model(img)
            f_trig = feature_fn(model, trig_img)
            f_trig = F.normalize(f_trig, dim=-1)
            features.append(f_trig)
            labels.append(torch.ones_like(label))

    return torch.cat(features, dim=0), torch.cat(labels, dim=0), trigger_batch_num


def compute_backdoor_axis(whitened_mixed: torch.Tensor) -> torch.Tensor:
    x = whitened_mixed - whitened_mixed.mean(dim=0, keepdim=True)
    x64 = x.double()
    try:
        _u, _s, vh = torch.linalg.svd(x64, full_matrices=False)
        axis = vh[0]
    except RuntimeError:
        cov = cov_centered(x64)
        _e, v = torch.linalg.eigh(cov)
        axis = v[:, -1]
    axis = axis.to(whitened_mixed.dtype)
    axis = axis / (torch.norm(axis) + 1e-12)
    return axis


def evaluate_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    trusted_scores: np.ndarray,
    threshold_percentile: float = 99.0,
) -> Dict[str, object]:
    threshold = float(np.percentile(trusted_scores, threshold_percentile))
    detections = scores > threshold
    cm = confusion_matrix(labels, detections, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def _safe_div(a, b):
        return float(a / b) if b else float("nan")

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)
    fnr = _safe_div(fn, fn + tp)

    auc_roc = float(roc_auc_score(labels, scores))
    pr_prec, pr_rec, _ = precision_recall_curve(labels, scores)
    auprc = float(sk_auc(pr_rec, pr_prec))
    ap = float(average_precision_score(labels, scores))

    report = classification_report(labels, detections, digits=4, output_dict=True)

    return {
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "auc_roc": auc_roc,
        "auprc": auprc,
        "ap": ap,
        "report": report,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_metrics(path: Path, payload: Dict[str, object]) -> None:
    serializable = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.detach().cpu().tolist()
        else:
            serializable[k] = v
    save_json(path, serializable)


def plot_roc_curve(path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = sk_auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Backdoor Detection")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_kde_scores(
    path: Path,
    clean_scores: np.ndarray,
    poison_scores: np.ndarray,
    trusted_scores: np.ndarray,
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 5))
    if clean_scores.size > 0:
        sns.kdeplot(clean_scores, fill=True, color="blue", alpha=0.35, label="Clean", clip=(0, None))
    if poison_scores.size > 0:
        sns.kdeplot(poison_scores, fill=True, color="red", alpha=0.35, label="Poisoned", clip=(0, None))
    if trusted_scores.size > 0:
        sns.kdeplot(trusted_scores, fill=False, color="black", linewidth=2, label="Trusted", clip=(0, None))
    plt.axvline(threshold, color="green", linestyle="--", linewidth=2, label="Threshold")
    plt.xlabel("|projection| onto backdoor axis")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cov_heatmaps(path: Path, cov_a: torch.Tensor, cov_b: torch.Tensor, k: int = 50) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    k = int(min(k, cov_a.shape[0], cov_a.shape[1]))
    cov_a_np = cov_a.detach().cpu().numpy()[:k, :k]
    cov_b_np = cov_b.detach().cpu().numpy()[:k, :k]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cov_a_np, cmap="viridis", ax=axes[0])
    axes[0].set_title("Covariance (before)")
    sns.heatmap(cov_b_np, cmap="viridis", ax=axes[1])
    axes[1].set_title("Covariance (after whitening)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scree(path: Path, eig_clean: np.ndarray, eig_poison: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(eig_clean, "b-", label="Clean Eigenvalues")
    plt.plot(eig_poison, "r-", label="Poison Eigenvalues")
    plt.title("Whitened Eigenvalues Comparison")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def maybe_plot_scatter_matrix(path: Path, clean_proj: np.ndarray, poison_proj: np.ndarray) -> None:
    try:
        import plotly.express as px
    except Exception:
        return
    import pandas as pd

    cols = ["PC1", "PC2", "PC3", "PC4"]
    df_clean = pd.DataFrame(clean_proj, columns=cols)
    df_clean["species"] = "clean"
    df_poison = pd.DataFrame(poison_proj, columns=cols)
    df_poison["species"] = "trigger"
    df_plot = pd.concat([df_clean, df_poison], ignore_index=True)
    fig = px.scatter_matrix(
        df_plot,
        dimensions=cols,
        color="species",
        title="PCA Scatter Matrix after Whitening",
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html(str(path))