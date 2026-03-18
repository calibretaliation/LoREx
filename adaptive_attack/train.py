import argparse
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

try:
    from sklearn.neural_network import MLPClassifier

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# Allow importing BadEncoder and this package regardless of cwd
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
BADENCODER_ROOT = PROJECT_ROOT / "BadEncoder"

for path in (PROJECT_ROOT, BADENCODER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import get_shadow_dataset, get_dataset_evaluation  # type: ignore
from datasets.backdoor_dataset import ReferenceImg  # type: ignore
from models import get_encoder_architecture_usage  # type: ignore

from adaptive_attack.models import UNetGenerator
from adaptive_attack.utils import (
    compute_low_variance_subspace,
    compute_target_embedding,
    encode_features,
    select_encoder_submodule,
    set_seed,
)


def _infer_reference_label_from_npz(reference_file: str) -> Optional[int]:
    if not reference_file:
        return None
    try:
        arr = np.load(reference_file)
        if "y" not in arr:
            return None
        y = arr["y"]
        if y.size == 0:
            return None
        y = np.asarray(y).reshape(-1)
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[int(np.argmax(counts))])
    except Exception:
        return None


def _dataset_data_dir(dataset: str) -> str:
    return str(PROJECT_ROOT / f"data/{dataset}/") + "/"


def _make_eval_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.dataset = dataset
    eval_args.data_dir = _dataset_data_dir(dataset)
    return eval_args


@torch.no_grad()
def compute_mean_embedding(
    encoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    encoder.eval()
    feats = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, desc="Mean embedding", leave=False)):
            images = batch[0].to(device, non_blocking=True)
            z = encode_features(encoder, images)
            feats.append(z)
            if max_batches is not None and max_batches > 0 and (idx + 1) >= max_batches:
                break
    if not feats:
        raise RuntimeError("Mean embedding loader is empty")
    mean = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)
    return F.normalize(mean, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive BadEncoder attack with UNet trigger generator")
    parser.add_argument("--batch_size", default=128, type=int, help="Mini-batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for student and generator")
    parser.add_argument("--lambda_clean", default=1.0, type=float, help="Weight for clean consistency loss")
    parser.add_argument("--lambda_target", default=1.0, type=float, help="Weight for backdoor target loss")
    parser.add_argument("--lambda_evasion", default=1.0, type=float, help="Weight for whitening evasion loss")
    parser.add_argument("--lambda_stealth", default=1e-3, type=float, help="Weight for pixel-level stealth loss")
    parser.add_argument("--eig_k", default=128, type=int, help="Number of low-variance eigenvectors")
    parser.add_argument(
        "--max_eigen_batches",
        default=200,
        type=int,
        help="Limit batches for eigen computation (use -1 for full dataset)",
    )
    parser.add_argument("--save_freq", default=50, type=int, help="Checkpoint frequency in epochs")
    parser.add_argument("--results_dir", default="./adaptive_attack_results", type=str, help="Directory to store checkpoints")
    # Checkpoints
    # teacher_ckpt should be truly clean/frozen; student_init_ckpt initializes the trainable student.
    # pretrained_encoder is kept for backward compatibility (acts as teacher_ckpt+student_init_ckpt).
    parser.add_argument("--teacher_ckpt", default="", type=str, help="Path to CLEAN teacher encoder checkpoint (frozen)")
    parser.add_argument("--student_init_ckpt", default="", type=str, help="Path to student init checkpoint (defaults to teacher_ckpt)")
    parser.add_argument("--pretrained_encoder", default="", type=str, help="[deprecated] Alias for teacher_ckpt + student_init_ckpt")
    parser.add_argument("--shadow_dataset", default="cifar10", type=str, help="Shadow dataset name")
    parser.add_argument("--reference_file", default="", type=str, help="Path to reference inputs (npz)")
    parser.add_argument("--trigger_file", default="", type=str, help="Path to trigger file (npz) required by BadEncoder dataset loader")
    parser.add_argument("--encoder_usage_info", default="cifar10", type=str, help="Encoder usage identifier: cifar10 | stl10 | imagenet | CLIP")
    parser.add_argument("--reference_label", default=0, type=int, help="Target label used by evaluation helpers")
    parser.add_argument("--knn_k", default=200, type=int, help="k for KNN monitor (evaluation)")
    parser.add_argument("--knn_t", default=0.5, type=float, help="temperature for KNN monitor")
    parser.add_argument("--seed", default=100, type=int, help="Random seed")
    parser.add_argument("--gpu", default="0", type=str, help="CUDA_VISIBLE_DEVICES setting")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument(
        "--max_train_batches",
        default=-1,
        type=int,
        help="Limit training batches per epoch for quick debugging (-1 for full epoch)",
    )
    parser.add_argument("--downstream_dataset", default="gtsrb", type=str, help="Downstream dataset for eval")
    parser.add_argument("--downstream_batch_size", default=256, type=int, help="Batch size for downstream eval feature extraction")
    parser.add_argument("--downstream_eval_freq", default=50, type=int, help="Epoch frequency for downstream eval")
    parser.add_argument(
        "--downstream_eval_mode",
        default="generator",
        choices=["generator", "badencoder_patch"],
        type=str,
        help="How to build backdoor inputs for downstream ASR: 'generator' uses the learned UNet; 'badencoder_patch' uses trigger_file.",
    )

    # Datasets for training/evasion statistics
    parser.add_argument(
        "--target_dataset",
        default="gtsrb",
        type=str,
        help="Dataset used to define the target direction/mean for the backdoor objective (usually the downstream dataset)",
    )
    parser.add_argument(
        "--evasion_dataset",
        default="gtsrb",
        type=str,
        help="Dataset used to compute whitening-evasion low-variance subspace (should match defense distribution)",
    )
    parser.add_argument(
        "--max_target_batches",
        default=200,
        type=int,
        help="Limit batches for target mean/direction computation (-1 for full dataset)",
    )
    parser.add_argument(
        "--use_residual_target",
        action="store_true",
        help="Use residual-to-mean target objective on target_dataset (recommended for downstream ASR)",
    )
    return parser.parse_args()


def _resolve_ckpts(args: argparse.Namespace) -> Tuple[str, str]:
    teacher_ckpt = getattr(args, "teacher_ckpt", "") or ""
    student_init_ckpt = getattr(args, "student_init_ckpt", "") or ""

    # Backward compatibility: pretrained_encoder acts as both.
    if getattr(args, "pretrained_encoder", ""):
        if not teacher_ckpt:
            teacher_ckpt = args.pretrained_encoder
        if not student_init_ckpt:
            student_init_ckpt = args.pretrained_encoder

    if not student_init_ckpt:
        student_init_ckpt = teacher_ckpt

    return teacher_ckpt, student_init_ckpt


def build_models(args: argparse.Namespace, device: torch.device):
    teacher_model = get_encoder_architecture_usage(args).to(device)
    student_model = get_encoder_architecture_usage(args).to(device)

    teacher_ckpt, student_init_ckpt = _resolve_ckpts(args)

    if teacher_ckpt:
        print(f"Loading teacher encoder from {teacher_ckpt}")
        checkpoint = torch.load(teacher_ckpt, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if args.encoder_usage_info in ["cifar10", "stl10"]:
            teacher_model.load_state_dict(state_dict)
        elif args.encoder_usage_info in ["imagenet", "CLIP"]:
            teacher_model.visual.load_state_dict(state_dict)
        else:
            raise NotImplementedError(f"Unsupported encoder_usage_info: {args.encoder_usage_info}")

    if student_init_ckpt:
        print(f"Initializing student encoder from {student_init_ckpt}")
        checkpoint = torch.load(student_init_ckpt, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if args.encoder_usage_info in ["cifar10", "stl10"]:
            student_model.load_state_dict(state_dict)
        elif args.encoder_usage_info in ["imagenet", "CLIP"]:
            student_model.visual.load_state_dict(state_dict)
        else:
            raise NotImplementedError(f"Unsupported encoder_usage_info: {args.encoder_usage_info}")

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    teacher_encoder = select_encoder_submodule(teacher_model, args.encoder_usage_info)
    student_encoder = select_encoder_submodule(student_model, args.encoder_usage_info)

    generator = UNetGenerator()
    generator.to(device)

    return teacher_encoder, student_encoder, generator, student_model


def train_one_epoch(
    epoch: int,
    student_encoder: nn.Module,
    teacher_encoder: nn.Module,
    generator: UNetGenerator,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_emb: torch.Tensor,
    target_mean: Optional[torch.Tensor],
    target_dir: Optional[torch.Tensor],
    u_low: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    target_train_loader: Optional[DataLoader] = None,
):
    def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(x, y, dim=-1).mean()

    student_encoder.train()
    generator.train()

    total_loss, total_num = 0.0, 0
    metrics = {
        "clean": 0.0,
        "target": 0.0,
        "evasion": 0.0,
        "stealth": 0.0,
    }

    target_iter: Optional[Iterator] = iter(target_train_loader) if target_train_loader is not None else None

    progress = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, (img_shadow, *_) in enumerate(progress):
        img_shadow = img_shadow.to(device, non_blocking=True)

        # Clean/utility term on shadow data
        with torch.no_grad():
            z_clean_teacher = encode_features(teacher_encoder, img_shadow)
        z_clean_student_shadow = encode_features(student_encoder, img_shadow)
        l_clean = cosine_loss(z_clean_student_shadow, z_clean_teacher)

        # Backdoor/evasion terms on target dataset (defaults to shadow if not provided)
        img_target = img_shadow
        if target_iter is not None:
            try:
                batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                batch = next(target_iter)
            img_target = batch[0].to(device, non_blocking=True)

        poisoned, _ = generator(img_target)
        z_clean_student = encode_features(student_encoder, img_target)
        z_poison_student = encode_features(student_encoder, poisoned)

        trigger_signal = z_poison_student - z_clean_student
        proj_low = torch.matmul(trigger_signal, u_low)

        if getattr(args, "use_residual_target", False):
            if target_mean is None or target_dir is None:
                raise RuntimeError("Residual target mode enabled but target_mean/target_dir are missing")
            resid = z_poison_student - target_mean.expand_as(z_poison_student)
            resid = F.normalize(resid, dim=-1)
            l_target = 1.0 - F.cosine_similarity(resid, target_dir.expand_as(resid), dim=-1).mean()
        else:
            l_target = cosine_loss(z_poison_student, target_emb.expand_as(z_poison_student))

        l_evasion = proj_low.norm(p=2, dim=1).mean()
        l_stealth = (poisoned - img_target).pow(2).mean()

        loss = (
            args.lambda_clean * l_clean
            + args.lambda_target * l_target
            + args.lambda_evasion * l_evasion
            + args.lambda_stealth * l_stealth
        )

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(student_encoder.parameters()) + list(generator.parameters()), args.max_grad_norm
            )
        optimizer.step()

        batch_size = img_shadow.size(0)
        total_num += batch_size
        total_loss += loss.item() * batch_size
        metrics["clean"] += l_clean.item() * batch_size
        metrics["target"] += l_target.item() * batch_size
        metrics["evasion"] += l_evasion.item() * batch_size
        metrics["stealth"] += l_stealth.item() * batch_size

        progress.set_postfix(
            loss=total_loss / total_num,
            clean=metrics["clean"] / total_num,
            target=metrics["target"] / total_num,
            evasion=metrics["evasion"] / total_num,
            stealth=metrics["stealth"] / total_num,
        )

        if args.max_train_batches > 0 and (step + 1) >= args.max_train_batches:
            break

    for key in metrics:
        metrics[key] /= max(total_num, 1)

    return total_loss / max(total_num, 1), metrics


def extract_features(encoder: nn.Module, data_loader: DataLoader, device: torch.device):
    feats, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Feature extracting", leave=False):
            data = data.to(device, non_blocking=True)
            feat = encode_features(encoder, data)
            feats.append(feat.cpu())
            labels.append(target)
    if not feats:
        return None, None
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels


def extract_features_with_generator(
    encoder: nn.Module,
    generator: UNetGenerator,
    data_loader: DataLoader,
    device: torch.device,
):
    feats, labels = [], []
    encoder.eval()
    generator.eval()
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Feature extracting (generator)", leave=False):
            data = data.to(device, non_blocking=True)
            poisoned, _ = generator(data)
            feat = encode_features(encoder, poisoned)
            feats.append(feat.cpu())
            labels.append(target)
    if not feats:
        return None, None
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels


def evaluate_downstream_mlp(
    student_encoder: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    generator: Optional[UNetGenerator] = None,
):
    if not HAVE_SKLEARN:
        print("sklearn not available; skipping downstream MLP evaluation")
        return None

    eval_args = argparse.Namespace(**vars(args))
    eval_args.dataset = args.downstream_dataset
    eval_args.data_dir = str(PROJECT_ROOT / f"data/{args.downstream_dataset}/") + "/"

    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(eval_args)
    if train_data is None or test_data_clean is None:
        print("Downstream datasets not available; skipping evaluation")
        return None

    train_loader = DataLoader(train_data, batch_size=args.downstream_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_clean_loader = DataLoader(test_data_clean, batch_size=args.downstream_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_bd_loader = None
    if args.downstream_eval_mode == "badencoder_patch":
        if test_data_backdoor is None:
            print("Backdoor test set not available; skipping evaluation")
            return None
        test_bd_loader = DataLoader(
            test_data_backdoor,
            batch_size=args.downstream_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    X_train, y_train = extract_features(student_encoder, train_loader, device)
    X_clean, y_clean = extract_features(student_encoder, test_clean_loader, device)
    if args.downstream_eval_mode == "generator":
        if generator is None:
            print("No generator provided; skipping generator-based downstream ASR")
            return None
        X_bd, _ = extract_features_with_generator(student_encoder, generator, test_clean_loader, device)
    else:
        if test_bd_loader is None:
            print("Backdoor loader missing; skipping evaluation")
            return None
        X_bd, _ = extract_features(student_encoder, test_bd_loader, device)

    if X_train is None or X_clean is None or X_bd is None:
        print("Feature extraction failed; skipping downstream eval")
        return None

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=50, random_state=args.seed)
    clf.fit(X_train, y_train)

    benign_acc = clf.score(X_clean, y_clean)
    bd_preds = clf.predict(X_bd)
    asr = float(np.mean(bd_preds == eval_args.reference_label))

    inferred_ref = _infer_reference_label_from_npz(getattr(eval_args, "reference_file", ""))
    if inferred_ref is not None and inferred_ref != eval_args.reference_label:
        inferred_asr = float(np.mean(bd_preds == inferred_ref))
        print(
            f"[warn] reference_label={eval_args.reference_label} but reference_file suggests {inferred_ref}; "
            f"ASR@{inferred_ref}={inferred_asr*100:.2f}%"
        )

    top = Counter(map(int, bd_preds)).most_common(5)
    top_str = ", ".join([f"{lbl}:{100*c/len(bd_preds):.2f}%" for lbl, c in top])
    print(f"Poisoned prediction top-5: {top_str}")

    print(
        f"Downstream MLP ({args.downstream_dataset}, eval_mode={args.downstream_eval_mode}): benign_acc={benign_acc*100:.2f}%, ASR={asr*100:.2f}%"
    )

    return {"benign_acc": benign_acc, "asr": asr}


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.data_dir = str(PROJECT_ROOT / f"data/{args.shadow_dataset.split('_')[0]}/") + "/"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print("Arguments:", args)

    shadow_data, _, _, _ = get_shadow_dataset(args)
    train_loader = DataLoader(
        shadow_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Build target dataset loader (for backdoor objective) and evasion dataset loader (for whitening subspace)
    target_args = _make_eval_args(args, args.target_dataset)
    target_dataset, target_train_data, _, _ = get_dataset_evaluation(target_args)
    target_train_loader = DataLoader(
        target_train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    evasion_args = _make_eval_args(args, args.evasion_dataset)
    _, evasion_train_data, _, _ = get_dataset_evaluation(evasion_args)
    evasion_loader = DataLoader(
        evasion_train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher_encoder, student_encoder, generator, student_model = build_models(args, device)
    teacher_encoder.eval()

    # Target embedding and (optional) residual target direction are computed on target_dataset
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Computing target embedding on target_dataset={args.target_dataset}...")
    target_emb = compute_target_embedding(teacher_encoder, target_loader, device)

    target_mean_batches = None if args.max_target_batches < 0 else args.max_target_batches
    mean_loader = DataLoader(
        target_train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Computing dataset mean embedding on target_dataset={args.target_dataset}...")
    target_mean = compute_mean_embedding(teacher_encoder, mean_loader, device, max_batches=target_mean_batches)

    target_dir = None
    if getattr(args, "use_residual_target", False):
        target_dir = F.normalize(target_emb - target_mean, dim=-1)
        print("Using residual target objective (target_dir = normalize(target_emb - mean_emb))")
    else:
        print("Using direct target embedding objective (cosine to target_emb)")

    eigen_batches = None if args.max_eigen_batches < 0 else args.max_eigen_batches
    print(f"Computing low-variance subspace for whitening evasion on evasion_dataset={args.evasion_dataset}...")
    u_low = compute_low_variance_subspace(
        teacher_encoder,
        evasion_loader,
        device,
        eig_k=args.eig_k,
        max_batches=eigen_batches,
    )

    optimizer = torch.optim.Adam(
        list(student_encoder.parameters()) + list(generator.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    downstream_metrics = None
    for epoch in range(1, args.epochs + 1):
        print("=" * 50)
        epoch_loss, metric_dict = train_one_epoch(
            epoch,
            student_encoder,
            teacher_encoder,
            generator,
            train_loader,
            optimizer,
            target_emb,
            target_mean,
            target_dir,
            u_low,
            device,
            args,
            target_train_loader=target_train_loader,
        )
        print(
            f"Epoch {epoch}: loss={epoch_loss:.4f}, clean={metric_dict['clean']:.4f}, "
            f"target={metric_dict['target']:.4f}, evasion={metric_dict['evasion']:.4f}, "
            f"stealth={metric_dict['stealth']:.6f}"
        )

        if args.downstream_eval_freq > 0 and (epoch % args.downstream_eval_freq == 0 or epoch == args.epochs):
            downstream_metrics = evaluate_downstream_mlp(student_encoder, args, device, generator=generator)

        if args.save_freq > 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            ckpt_path = Path(args.results_dir) / f"adaptive_attack_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": student_model.state_dict(),
                    "generator_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "target_embedding": target_emb.detach().cpu(),
                    "target_mean": target_mean.detach().cpu(),
                    "target_dir": None if target_dir is None else target_dir.detach().cpu(),
                    "u_low": u_low.detach().cpu(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if downstream_metrics is None:
        downstream_metrics = evaluate_downstream_mlp(student_encoder, args, device, generator=generator)

    if downstream_metrics is not None:
        print(
            f"Final downstream ({args.downstream_dataset}) benign_acc={downstream_metrics['benign_acc']*100:.2f}%, "
            f"ASR={downstream_metrics['asr']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
