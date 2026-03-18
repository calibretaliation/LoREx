"""LoREx — Low-Rank Exposure: whitening-based SSL backdoor detection.

Sub-modules:
    whitening  — ZCA whitening matrix and whiten_and_pca helper
    scoring    — MS-Spectral scoring (spectral_score, ms_spectral, K_RANGE)
    metrics    — Detection metrics (full_metrics, compute_auc, evaluate_detection)
    features   — Feature extraction from SSL encoders
    viz        — Paper-quality figures (ROC, score dist, per-K heatmap)
"""

from .whitening import cov_centered, compute_whitening_matrix, whiten_and_pca
from .scoring import spectral_score, ms_spectral, K_RANGE
from .metrics import compute_auc, full_metrics, evaluate_detection, save_json, save_metrics
from .features import (
    extract_features,
    extract_clean_and_poison_features_pair,
    extract_clean_and_poison_features_trigger,
)

__all__ = [
    # whitening
    "cov_centered",
    "compute_whitening_matrix",
    "whiten_and_pca",
    # scoring
    "spectral_score",
    "ms_spectral",
    "K_RANGE",
    # metrics
    "compute_auc",
    "full_metrics",
    "evaluate_detection",
    "save_json",
    "save_metrics",
    # features
    "extract_features",
    "extract_clean_and_poison_features_pair",
    "extract_clean_and_poison_features_trigger",
]
