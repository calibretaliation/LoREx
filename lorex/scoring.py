"""MS-Spectral scoring for LoREx.

Core detection formula (per sample z in whitened space):

    S_K(z) = sum_{k=1}^K (v_k' z)^2 / lambda_k

where v_k / lambda_k are the k-th eigenvector / eigenvalue of the whitened
clean covariance (sorted largest-first).

After computing S_K for each K in K_RANGE, scores are z-normalized using the
trusted-set statistics for that K, then aggregated across K via max.

Reference: tmp/ms_spectral_final.py
"""

import numpy as np

# Default K values — covers a wide spectral range without being redundant.
K_RANGE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def spectral_score(
    z: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    K: int,
) -> np.ndarray:
    """Compute S_K(z) = sum_{k=1}^K (v_k' z)^2 / lambda_k.

    Args:
        z: whitened feature array, shape (n, d).
        eigvecs: eigenvectors, shape (d, d), columns sorted largest-first.
        eigvals: eigenvalues, shape (d,), sorted largest-first.
        K: number of top eigenvectors to use.

    Returns:
        Scores, shape (n,).
    """
    V = eigvecs[:, :K]       # (d, K)
    lam = eigvals[:K]        # (K,)
    return np.sum((z @ V) ** 2 / lam[None, :], axis=1)


def ms_spectral(
    tw: np.ndarray,
    cw: np.ndarray,
    pw: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    k_range: list = None,
    agg: str = "max",
) -> tuple:
    """Multi-scale spectral detector.

    For each K in k_range:
      1. Compute S_K for trusted (tw), test-clean (cw), test-poison (pw).
      2. Z-normalize cw/pw scores using trusted mean/std.

    Then aggregate the z-normalized scores across K.

    Args:
        tw: whitened trusted features, shape (n_trusted, d).
        cw: whitened test-clean features, shape (n_clean, d).
        pw: whitened test-poison features, shape (n_poison, d).
        eigvecs: eigenvectors from whitened trusted covariance, (d, d).
        eigvals: eigenvalues, (d,), sorted largest-first.
        k_range: list of K values (default: K_RANGE).
        agg: aggregation strategy — "max", "mean", "median", "p90", "top3mean".

    Returns:
        (clean_scores, poison_scores, per_k_clean, per_k_poison)
        where per_k_* are (len(k_range), n) arrays of per-K z-scores.
    """
    if k_range is None:
        k_range = K_RANGE

    all_c, all_p = [], []
    for K in k_range:
        st = spectral_score(tw, eigvecs, eigvals, K)
        sc = spectral_score(cw, eigvecs, eigvals, K)
        sp = spectral_score(pw, eigvecs, eigvals, K)
        mu, std = np.mean(st), np.std(st) + 1e-10
        all_c.append((sc - mu) / std)
        all_p.append((sp - mu) / std)

    arr_c = np.array(all_c)   # (len(k_range), n_clean)
    arr_p = np.array(all_p)   # (len(k_range), n_poison)

    if agg == "max":
        return arr_c.max(0), arr_p.max(0), arr_c, arr_p
    elif agg == "mean":
        return arr_c.mean(0), arr_p.mean(0), arr_c, arr_p
    elif agg == "median":
        return np.median(arr_c, 0), np.median(arr_p, 0), arr_c, arr_p
    elif agg == "p90":
        return np.percentile(arr_c, 90, axis=0), np.percentile(arr_p, 90, axis=0), arr_c, arr_p
    elif agg == "top3mean":
        top_c = np.sort(arr_c, axis=0)[-3:].mean(0)
        top_p = np.sort(arr_p, axis=0)[-3:].mean(0)
        return top_c, top_p, arr_c, arr_p
    else:
        raise ValueError(f"Unknown aggregation: {agg!r}. Choose from max/mean/median/p90/top3mean.")
