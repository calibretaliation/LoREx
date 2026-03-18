"""Visualization utilities for LoREx — paper-quality figures only.

Three figure types matching the MS-Spectral paper figures:
  - Fig 1: ROC curves per attack (with low-FPR inset)
  - Fig 2: Score distributions (clean vs poison histogram)
  - Fig 3: Per-K AUC heatmap (K × attack grid)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Default colour / marker scheme per attack
_ATK_COLORS = {
    "badencoder": "#1f77b4",
    "drupe":      "#ff7f0e",
    "inactive":   "#2ca02c",
    "badclip":    "#d62728",
}
_ATK_MARKERS = {
    "badencoder": "o",
    "drupe":      "s",
    "inactive":   "D",
    "badclip":    "^",
}
_ATK_LABELS = {
    "badencoder": "BadEncoder",
    "drupe":      "DRUPE",
    "inactive":   "INACTIVE",
    "badclip":    "BadCLIP",
}


def _atk_color(atk: str) -> str:
    return _ATK_COLORS.get(atk, "#7f7f7f")


def _atk_label(atk: str) -> str:
    return _ATK_LABELS.get(atk, atk)


def plot_roc_curves(path: Path, results: dict) -> None:
    """Multi-attack ROC curves with low-FPR inset (Fig 1 style).

    Args:
        path: output file path (.png).
        results: dict mapping attack_name -> full_metrics() output dict,
                 each containing 'fpr', 'tpr', 'auc', 'tpr@1%', 'tpr@5%'.
    """
    attacks = list(results.keys())
    n = len(attacks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, atk in zip(axes, attacks):
        m = results[atk]
        color = _atk_color(atk)
        ax.plot(m["fpr"], m["tpr"], color=color, lw=2.5,
                label=f'AUC = {m["auc"]:.4f}')
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(_atk_label(atk), fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # Low-FPR inset
        axins = ax.inset_axes([0.35, 0.15, 0.55, 0.45])
        axins.plot(m["fpr"], m["tpr"], color=color, lw=2)
        axins.plot([0, 0.1], [0, 0.1], "k--", lw=0.5)
        axins.set_xlim([-0.002, 0.12])
        axins.set_ylim([-0.02, 1.02])
        axins.set_xlabel("FPR", fontsize=8)
        axins.set_ylabel("TPR", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3)
        axins.axvline(0.01, color="gray", ls=":", lw=0.7)
        axins.axvline(0.05, color="gray", ls="--", lw=0.7)
        axins.plot(0.01, m["tpr@1%"], "r*", ms=10, zorder=5)
        axins.plot(0.05, m["tpr@5%"], "r^", ms=8, zorder=5)
        axins.annotate(f'@1%={m["tpr@1%"]:.2f}', (0.01, m["tpr@1%"]),
                       fontsize=7, xytext=(5, -15), textcoords="offset points",
                       color="red")
        axins.annotate(f'@5%={m["tpr@5%"]:.2f}', (0.05, m["tpr@5%"]),
                       fontsize=7, xytext=(5, 5), textcoords="offset points",
                       color="red")

    fig.suptitle("MS-Spectral: ROC Curves (max aggregation)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(path: Path, results: dict) -> None:
    """Score histograms: clean vs poison per attack (Fig 2 style).

    Args:
        path: output file path (.png).
        results: dict mapping attack_name -> full_metrics() output,
                 each containing 'cs' (clean scores) and 'ps' (poison scores).
    """
    attacks = list(results.keys())
    n = len(attacks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, atk in zip(axes, attacks):
        m = results[atk]
        cs, ps = m["cs"], m["ps"]
        lo = min(cs.min(), ps.min()) - 0.5
        hi = max(cs.max(), ps.max()) + 0.5
        bins = np.linspace(lo, hi, 60)
        ax.hist(cs, bins, alpha=0.6, color="steelblue", edgecolor="navy",
                label=f"Clean (n={len(cs)})", density=True)
        ax.hist(ps, bins, alpha=0.6, color="crimson", edgecolor="darkred",
                label=f"Poison (n={len(ps)})", density=True)
        ax.axvline(np.mean(cs), color="steelblue", ls="--", lw=1.5)
        ax.axvline(np.mean(ps), color="crimson", ls="--", lw=1.5)
        ax.set_xlabel("MS-Spectral z-score (max)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{_atk_label(atk)}  (AUC={m['auc']:.4f})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("MS-Spectral: Score Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_k_auc_heatmap(
    path: Path,
    per_k_aucs: dict,
    k_range: list,
    attacks: list,
) -> None:
    """Heatmap of AUC vs K (single-scale) for each attack (Fig 3 style).

    Args:
        path: output file path (.png).
        per_k_aucs: dict mapping attack_name -> list of per-K AUC values
                    (one per entry in k_range).
        k_range: list of K values used.
        attacks: list of attack names (row order).
    """
    heatdata = np.array([per_k_aucs[atk] for atk in attacks])
    fig, ax = plt.subplots(figsize=(max(8, len(k_range) * 0.8), max(3, len(attacks))))
    im = ax.imshow(heatdata, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([_atk_label(a) for a in attacks], fontsize=11)
    ax.set_xticks(range(len(k_range)))
    ax.set_xticklabels([str(k) for k in k_range], fontsize=9)
    ax.set_xlabel("K (spectral components)", fontsize=11)
    ax.set_title("Per-K AUC (single scale, no aggregation)", fontsize=13)
    for y in range(len(attacks)):
        for x in range(len(k_range)):
            val = heatdata[y, x]
            color = "white" if val < 0.7 else "black"
            ax.text(x, y, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
