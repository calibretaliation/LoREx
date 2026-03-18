"""LoREx — MS-Spectral backdoor detection for SSL encoders.

Pipeline:
  1. Load SSL encoder (via attacks.py) and dataset (via dataset.py).
  2. Extract L2-normalized features from the trusted clean set.
  3. Extract separate clean and poisoned features from the test set.
  4. Whiten using trusted set statistics, then PCA-decompose.
  5. Score every test sample with MS-Spectral:
        S_K(z) = sum_{k=1}^K (v_k' z)^2 / lambda_k
     Z-normalize per K against trusted-set statistics, aggregate with max.
  6. Report AUC, TPR@1%/5%/10% FPR, and save figures + JSON.

Reference scoring implementation: tmp/ms_spectral_final.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from attacks import build_attack_from_args
from dataset import build_dataset_from_args
from lorex import (
    whiten_and_pca,
    ms_spectral,
    full_metrics,
    compute_auc,
    extract_features,
    extract_clean_and_poison_features_pair,
    extract_clean_and_poison_features_trigger,
    save_metrics,
    K_RANGE,
)
from lorex.viz import plot_roc_curves, plot_score_distributions, plot_per_k_auc_heatmap


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoREx MS-Spectral backdoor detection for SSL encoders"
    )

    # ── Attack / dataset ────────────────────────────────────────────────
    parser.add_argument(
        "--attack", type=str, required=True,
        choices=["badencoder", "badclip", "drupe", "inactive"],
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=[
            "cifar10_npz_pair",
            "badencoder_cifar10_pair",
            "imagenet_subset_pair",
            "cifar10_224_clean",
            "imagenet_clean",
        ],
    )

    # ── Detection hyperparameters ────────────────────────────────────────
    parser.add_argument(
        "--k_range", type=int, nargs="+", default=None,
        help=f"K values for MS-Spectral (default: {K_RANGE})",
    )
    parser.add_argument(
        "--agg", type=str, default="max",
        choices=["max", "mean", "median", "p90", "top3mean"],
        help="Aggregation strategy across K values (default: max)",
    )
    parser.add_argument(
        "--n_trusted", type=int, default=5000,
        help="Number of trusted clean samples for whitening + z-normalization",
    )
    parser.add_argument(
        "--n_test", type=int, default=200,
        help="Number of test samples per class (clean / poisoned)",
    )

    # ── Infrastructure ───────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap feature extraction at this many batches (debug)")

    # ── Model / data paths ───────────────────────────────────────────────
    parser.add_argument("--attack_ckpt_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--trigger_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10/")
    parser.add_argument("--unet_path", type=str, default="")

    # ── Attack-specific options ──────────────────────────────────────────
    parser.add_argument("--badencoder_usage_info", type=str, default="cifar10")
    parser.add_argument("--clip_name", type=str, default="RN50")
    parser.add_argument("--inactive_model_type", type=str, default="clip",
                        choices=["clip", "simclr"])
    parser.add_argument("--inactive_config_path", type=str, default="")
    parser.add_argument("--inactive_encoder_path", type=str, default="")

    # ── Dataset-specific options ─────────────────────────────────────────
    parser.add_argument("--imagenet_val_dir", type=str, default="")
    parser.add_argument("--labels_csv", type=str, default="")
    parser.add_argument("--classes_py", type=str, default="")
    parser.add_argument("--clean_subset_frac", type=float, default=0.1)
    parser.add_argument("--trigger_size", type=int, default=32)
    parser.add_argument("--trusted_frac", type=float, default=0.1)
    parser.add_argument("--no_augment", action="store_true")

    return parser.parse_args()


def main():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = parse_args()
    set_seed(args.seed)

    k_range = args.k_range if args.k_range else K_RANGE
    device = torch.device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    fig_dir = ensure_dir(output_dir / "figures")
    metrics_dir = ensure_dir(output_dir / "metrics")

    # ── Step 1: Build attack and dataset ────────────────────────────────
    attack = build_attack_from_args(args)
    dataset = build_dataset_from_args(args, attack)
    model = attack.model
    feature_fn = attack.feature_fn
    n_need = args.n_trusted + args.n_test

    print(f"[lorex] attack={attack.name}  dataset={dataset.name}  device={device}")
    print(f"[lorex] k_range={k_range}  agg={args.agg}")
    print(f"[lorex] n_trusted={args.n_trusted}  n_test={args.n_test}")

    # ── Step 2: Extract trusted clean features ───────────────────────────
    print("[lorex] extracting trusted clean features ...")
    all_trusted = extract_features(
        dataset.trusted_loader, model, device, feature_fn,
        max_batches=args.max_batches,
    )
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(all_trusted.shape[0])
    trusted = all_trusted[perm[: args.n_trusted]]
    print(f"[lorex] trusted: {trusted.shape}")

    # ── Step 3: Extract test clean + poison features ─────────────────────
    print("[lorex] extracting test features ...")
    if dataset.pair_loader:
        all_clean, all_poison = extract_clean_and_poison_features_pair(
            dataset.mixed_loader, model, device, feature_fn,
            max_samples=n_need + 128,
        )
    else:
        if attack.trigger_model is None:
            raise ValueError(
                "Non-pair datasets require a trigger model (e.g. INACTIVE UNet). "
                "Pass --unet_path."
            )
        all_clean, all_poison = extract_clean_and_poison_features_trigger(
            dataset.mixed_loader, model, attack.trigger_model, device, feature_fn,
            max_samples=n_need + 128,
        )

    perm_c = rng.permutation(all_clean.shape[0])
    perm_p = rng.permutation(all_poison.shape[0])
    tc = all_clean[perm_c[: args.n_test]]
    tp = all_poison[perm_p[: args.n_test]]
    print(f"[lorex] test_clean: {tc.shape}  test_poison: {tp.shape}")

    # ── Step 4: Whiten and PCA ───────────────────────────────────────────
    print("[lorex] whitening + PCA ...")
    tw, cw, pw, eigvals, eigvecs = whiten_and_pca(trusted, tc, tp)

    # ── Step 5: MS-Spectral scoring ──────────────────────────────────────
    print(f"[lorex] scoring with MS-Spectral (agg={args.agg}) ...")
    clean_scores, poison_scores, per_k_c, per_k_p = ms_spectral(
        tw, cw, pw, eigvecs, eigvals, k_range, agg=args.agg,
    )

    # ── Step 6: Metrics ──────────────────────────────────────────────────
    m = full_metrics(clean_scores, poison_scores)

    # Per-K AUC (for heatmap figure)
    from lorex.scoring import spectral_score
    per_k_aucs = {attack.name: []}
    for ki, K in enumerate(k_range):
        auc_k = compute_auc(per_k_c[ki], per_k_p[ki])
        per_k_aucs[attack.name].append(auc_k)

    # ── Step 7: Console output ───────────────────────────────────────────
    print()
    print(f"  {'Attack':<15s} │ {'AUC':>7s} │ {'TPR@1%':>7s} │ {'TPR@5%':>7s} │ {'TPR@10%':>8s}")
    print(f"  {'─'*15}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}")
    print(
        f"  {attack.name:<15s} │ {m['auc']:7.4f} │ {m['tpr@1%']:7.4f} │ "
        f"{m['tpr@5%']:7.4f} │ {m['tpr@10%']:8.4f}"
    )
    print()

    # ── Step 8: Save metrics ─────────────────────────────────────────────
    payload = {
        "attack": attack.name,
        "dataset": dataset.name,
        "device": str(device),
        "seed": args.seed,
        "k_range": k_range,
        "agg": args.agg,
        "n_trusted": trusted.shape[0],
        "n_test_clean": tc.shape[0],
        "n_test_poison": tp.shape[0],
        "auc": m["auc"],
        "tpr@1%": m["tpr@1%"],
        "tpr@5%": m["tpr@5%"],
        "tpr@10%": m["tpr@10%"],
        "per_k_aucs": {str(K): float(per_k_aucs[attack.name][i])
                       for i, K in enumerate(k_range)},
    }
    save_metrics(metrics_dir / "results.json", payload)
    print(f"[lorex] metrics saved to {metrics_dir / 'results.json'}")

    # ── Step 9: Figures ──────────────────────────────────────────────────
    results_for_viz = {attack.name: m}
    plot_roc_curves(fig_dir / "roc_curves.png", results_for_viz)
    plot_score_distributions(fig_dir / "score_distributions.png", results_for_viz)
    plot_per_k_auc_heatmap(
        fig_dir / "per_k_auc_heatmap.png",
        per_k_aucs, k_range, [attack.name],
    )
    print(f"[lorex] figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
