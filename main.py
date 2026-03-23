import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from attacks import build_attack_from_args
from dataset import build_dataset_from_args
from utils import (
    compute_backdoor_axis,
    compute_whitening_matrix,
    ensure_dir,
    evaluate_detection,
    extract_features,
    extract_mixed_features_from_pair_loader,
    extract_mixed_features_with_trigger_model,
    plot_cov_heatmaps,
    plot_kde_scores,
    plot_roc_curve,
    plot_scree,
    maybe_plot_scatter_matrix,
    save_metrics,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRex whitening defense reproduction")
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=["badencoder", "badclip", "drupe", "inactive"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10_npz_pair", "badencoder_cifar10_pair", "imagenet_subset_pair", "cifar10_224_clean"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./whitening_outputs")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_trusted_batches", type=int, default=None)
    parser.add_argument("--max_mixed_batches", type=int, default=None)
    parser.add_argument("--trigger_chance", type=float, default=0.01)
    parser.add_argument("--trigger_batch_num", type=int, default=None)
    parser.add_argument("--threshold_percentile", type=float, default=99.0)

    parser.add_argument("--attack_ckpt_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--trigger_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10/")
    parser.add_argument("--unet_path", type=str, default="")

    parser.add_argument("--badencoder_usage_info", type=str, default="cifar10")
    parser.add_argument("--clip_name", type=str, default="RN50")
    parser.add_argument("--inactive_model_type", type=str, default="clip", choices=["clip", "simclr"])
    parser.add_argument("--inactive_config_path", type=str, default="")
    parser.add_argument("--inactive_encoder_path", type=str, default="")

    parser.add_argument("--imagenet_val_dir", type=str, default="")
    parser.add_argument("--labels_csv", type=str, default="")
    parser.add_argument("--classes_py", type=str, default="")
    parser.add_argument("--clean_subset_frac", type=float, default=0.1)
    parser.add_argument("--trigger_size", type=int, default=32)

    parser.add_argument("--trusted_frac", type=float, default=0.1)
    parser.add_argument("--no_augment", action="store_true")

    parser.add_argument("--plot_scatter_matrix", action="store_true")
    parser.add_argument("--plot_heatmaps", action="store_true")
    parser.add_argument("--heatmap_k", type=int, default=50)

    return parser.parse_args()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(Path(args.output_dir))
    fig_dir = ensure_dir(output_dir / "figures")
    metrics_dir = ensure_dir(output_dir / "metrics")

    attack = build_attack_from_args(args)
    dataset = build_dataset_from_args(args, attack)

    device = torch.device(args.device)
    model = attack.model
    feature_fn = attack.feature_fn

    clean_bank = extract_features(
        dataset.trusted_loader,
        model,
        device,
        feature_fn,
        max_batches=args.max_trusted_batches,
    )
    mu_clean = torch.mean(clean_bank, dim=0)
    centered_clean = clean_bank - mu_clean
    whitening_matrix = compute_whitening_matrix(centered_clean, method="cholesky", ridge=1e-3)
    whitened_clean = centered_clean @ whitening_matrix

    if args.plot_heatmaps:
        cov_clean = torch.cov(centered_clean.T, correction=1)
        cov_whitened_clean = torch.cov(whitened_clean.T, correction=1)
        plot_cov_heatmaps(fig_dir / "cov_clean_vs_whitened.png", cov_clean, cov_whitened_clean, k=args.heatmap_k)

    if dataset.pair_loader:
        mixed_features, mixed_labels = extract_mixed_features_from_pair_loader(
            dataset.mixed_loader,
            model,
            device,
            feature_fn,
            trigger_chance=args.trigger_chance,
            max_batches=args.max_mixed_batches,
        )
        trigger_batches = None
    else:
        if attack.trigger_model is None:
            raise ValueError("Non-pair datasets require a trigger model (e.g., INACTIVE UNet)")
        mixed_features, mixed_labels, trigger_batches = extract_mixed_features_with_trigger_model(
            dataset.mixed_loader,
            model,
            attack.trigger_model,
            device,
            feature_fn,
            trigger_batch_num=args.trigger_batch_num,
            max_batches=args.max_mixed_batches,
            seed=args.seed,
        )

    centered_mixed = mixed_features - mu_clean
    whitened_mixed = centered_mixed @ whitening_matrix
    cov_whitened_mixed = torch.cov(whitened_mixed.T, correction=1)

    backdoor_axis = compute_backdoor_axis(whitened_mixed)
    projections = whitened_mixed @ backdoor_axis

    scores_np = torch.abs(projections).detach().cpu().numpy()
    labels_np = mixed_labels.detach().cpu().numpy().astype(np.int64)

    trusted_scores = torch.abs(whitened_clean @ backdoor_axis).detach().cpu().numpy()
    metrics = evaluate_detection(
        scores_np,
        labels_np,
        trusted_scores,
        threshold_percentile=args.threshold_percentile,
    )

    metrics_payload = {
        "attack": attack.name,
        "dataset": dataset.name,
        "device": str(device),
        "seed": args.seed,
        "trusted_shape": tuple(clean_bank.shape),
        "mixed_shape": tuple(mixed_features.shape),
        "trigger_batches": trigger_batches,
        **metrics,
    }
    save_metrics(metrics_dir / "metrics.json", metrics_payload)

    plot_roc_curve(fig_dir / "roc_curve.png", labels_np, scores_np)
    clean_scores = scores_np[labels_np == 0]
    poison_scores = scores_np[labels_np == 1]
    plot_kde_scores(
        fig_dir / "kde_scores.png",
        clean_scores,
        poison_scores,
        trusted_scores,
        metrics["threshold"],
    )

    whitened_clean_all = whitened_mixed[mixed_labels == 0]
    whitened_poison_all = whitened_mixed[mixed_labels == 1]
    if whitened_clean_all.shape[0] > 2 and whitened_poison_all.shape[0] > 2:
        cov_clean = torch.cov(whitened_clean_all.T, correction=1)
        cov_poison = torch.cov(whitened_poison_all.T, correction=1)
        clean_eigs = torch.linalg.eigvalsh(cov_clean).detach().cpu().numpy()[::-1]
        poison_eigs = torch.linalg.eigvalsh(cov_poison).detach().cpu().numpy()[::-1]
        plot_scree(fig_dir / "scree_plot.png", clean_eigs, poison_eigs)

    if args.plot_scatter_matrix:
        if whitened_clean_all.shape[0] > 0 and whitened_poison_all.shape[0] > 0:
            backdoor_axes = torch.linalg.eigh(cov_whitened_mixed)[1][:, -4:]
            clean_proj = (whitened_clean_all @ backdoor_axes).detach().cpu().numpy()
            poison_proj = (whitened_poison_all @ backdoor_axes).detach().cpu().numpy()
            maybe_plot_scatter_matrix(fig_dir / "scatter_matrix.html", clean_proj, poison_proj)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()