"""Detection metrics for LoREx.

Provides both the paper-style metrics (AUC, TPR@k%) and a CLI-compatible
evaluate_detection helper, plus JSON serialization utilities.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    auc as sk_auc,
)


def compute_auc(clean_scores: np.ndarray, poison_scores: np.ndarray) -> float:
    """ROC-AUC, automatically choosing the correct score direction."""
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(poison_scores))])
    s = np.concatenate([clean_scores, poison_scores])
    return float(max(roc_auc_score(y, s), roc_auc_score(y, -s)))


def full_metrics(clean_scores: np.ndarray, poison_scores: np.ndarray) -> dict:
    """Compute AUC, TPR@1%/5%/10% FPR, plus fpr/tpr arrays for plotting.

    Automatically flips score direction if needed so higher = more poisoned.

    Returns:
        dict with keys: auc, tpr@1%, tpr@5%, tpr@10%, fpr, tpr, cs, ps
    """
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(poison_scores))])
    s = np.concatenate([clean_scores, poison_scores])
    a1 = roc_auc_score(y, s)
    a2 = roc_auc_score(y, -s)
    if a2 > a1:
        s, clean_scores, poison_scores, auc = -s, -clean_scores, -poison_scores, a2
    else:
        auc = a1

    fpr, tpr, _ = roc_curve(y, s)
    t1  = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0.0
    t5  = tpr[np.where(fpr <= 0.05)[0][-1]] if np.any(fpr <= 0.05) else 0.0
    t10 = tpr[np.where(fpr <= 0.10)[0][-1]] if np.any(fpr <= 0.10) else 0.0

    return {
        "auc": float(auc),
        "tpr@1%": float(t1),
        "tpr@5%": float(t5),
        "tpr@10%": float(t10),
        "fpr": fpr,
        "tpr": tpr,
        "cs": clean_scores,
        "ps": poison_scores,
    }


def evaluate_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    trusted_scores: np.ndarray,
    threshold_percentile: float = 99.0,
) -> Dict[str, object]:
    """Threshold-based detection metrics (for CLI / compatibility use).

    Args:
        scores: detection scores for test samples.
        labels: ground-truth labels (0=clean, 1=poison).
        trusted_scores: scores on the trusted clean set (used to set threshold).
        threshold_percentile: percentile of trusted scores used as threshold.

    Returns:
        dict with threshold, confusion_matrix, accuracy, tpr, fpr, tnr, fnr,
        auc_roc, auprc, ap, report.
    """
    threshold = float(np.percentile(trusted_scores, threshold_percentile))
    detections = scores > threshold
    cm = confusion_matrix(labels, detections, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def _safe_div(a, b):
        return float(a / b) if b else float("nan")

    auc_roc = float(roc_auc_score(labels, scores))
    pr_prec, pr_rec, _ = precision_recall_curve(labels, scores)
    auprc = float(sk_auc(pr_rec, pr_prec))
    ap = float(average_precision_score(labels, scores))
    report = classification_report(labels, detections, digits=4, output_dict=True)

    return {
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "tpr": _safe_div(tp, tp + fn),
        "fpr": _safe_div(fp, fp + tn),
        "tnr": _safe_div(tn, tn + fp),
        "fnr": _safe_div(fn, fn + tp),
        "auc_roc": auc_roc,
        "auprc": auprc,
        "ap": ap,
        "report": report,
    }


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_metrics(path: Path, payload: dict) -> None:
    import torch
    serializable = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.detach().cpu().tolist()
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    save_json(path, serializable)
