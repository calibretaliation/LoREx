"""Whitening utilities for LoREx.

Provides covariance estimation, ZCA whitening matrix computation, and the
combined whiten-then-PCA helper used by the MS-Spectral pipeline.
"""

import torch
import numpy as np


def cov_centered(x: torch.Tensor) -> torch.Tensor:
    """Centered sample covariance in float64."""
    x64 = x.double()
    x64 = x64 - x64.mean(dim=0, keepdim=True)
    n = x64.shape[0]
    return (x64.T @ x64) / max(1, n - 1)


def compute_whitening_matrix(
    centered_clean: torch.Tensor,
    method: str = "cholesky",
    ridge: float = 1e-3,
    max_tries: int = 8,
) -> torch.Tensor:
    """ZCA whitening matrix W such that W^T W ≈ Σ^{-1}.

    Args:
        centered_clean: mean-centered clean features, shape (n, d).
        method: "cholesky" (default) or "eig".
        ridge: regularization fraction of mean eigenvalue.
        max_tries: number of Cholesky retry attempts with increasing ridge.

    Returns:
        W of shape (d, d) in the same dtype as centered_clean.
    """
    cov = cov_centered(centered_clean)
    d = cov.shape[0]
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    tr = torch.trace(cov)
    base = float(tr.item() / d)
    base = max(base, 1e-12)
    lam0 = ridge * base

    if method == "eig":
        cov_reg = cov + lam0 * eye
        e_vals, e_vecs = torch.linalg.eigh(cov_reg)
        inv_sqrt = torch.rsqrt(torch.clamp(e_vals, min=lam0))
        w = e_vecs @ torch.diag(inv_sqrt) @ e_vecs.T
        return w.to(centered_clean.dtype)

    for i in range(int(max_tries)):
        lam = lam0 * (10.0 ** i)
        try:
            cov_reg = cov + lam * eye
            l = torch.linalg.cholesky(cov_reg)
            inv_l = torch.linalg.solve_triangular(l, eye, upper=False)
            return inv_l.T.to(centered_clean.dtype)
        except RuntimeError:
            continue

    # Fallback: eigendecomposition
    cov_reg = cov + lam0 * (10.0 ** (max_tries - 1)) * eye
    e_vals, e_vecs = torch.linalg.eigh(cov_reg)
    inv_sqrt = torch.rsqrt(torch.clamp(e_vals, min=lam0))
    w = e_vecs @ torch.diag(inv_sqrt) @ e_vecs.T
    return w.to(centered_clean.dtype)


def whiten_and_pca(
    trusted: torch.Tensor,
    *others: torch.Tensor,
) -> tuple:
    """Whiten trusted features with Cholesky, then PCA-decompose.

    Returns (*whitened_arrays, eigvals, eigvecs) where eigvals/eigvecs are
    sorted descending (largest first) — matching the MS-Spectral convention.

    Args:
        trusted: trusted clean features (n_trusted, d).
        *others: additional feature matrices to whiten with the same W/mu.

    Returns:
        tuple of (whitened_trusted, *whitened_others, eigvals_np, eigvecs_np)
        where eigvals/eigvecs are numpy arrays, sorted largest-first.
    """
    mu = trusted.mean(dim=0)
    W = compute_whitening_matrix(trusted - mu, method="cholesky", ridge=1e-3)
    tw = ((trusted - mu) @ W).cpu().numpy()
    out = [tw]
    for o in others:
        out.append(((o - mu) @ W).cpu().numpy())
    cov = np.cov(tw.T)
    ev, evec = np.linalg.eigh(cov)
    idx = np.argsort(ev)[::-1]
    return (*out, ev[idx], evec[:, idx])
