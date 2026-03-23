# LoREx — Low-Rank Exposure

LoREx is a whitening-based backdoor detection method for self-supervised learning (SSL) encoders. It exploits the fact that backdoor triggers leave a low-rank statistical footprint in feature space: by whitening the trusted clean distribution and decomposing the residual covariance, poisoned samples stand out as anomalies along specific principal directions. Detection requires only the suspicious encoder, a small trusted clean dataset, and the test images — no clean reference model or downstream labels are needed.

---

## How It Works

Given a trusted set of clean images, LoREx:

1. **Extracts L2-normalized features** from the encoder for both the trusted set and test images.
2. **Whitens** the feature space using the trusted-set covariance, then **PCA-decomposes** the whitened covariance to get eigenvectors `v_k` and eigenvalues `λ_k` (sorted largest-first).
3. **Scores** each test sample `z` (in whitened space) with the **MS-Spectral formula**:

   ```
   S_K(z) = Σ_{k=1}^{K}  (v_k' z)² / λ_k
   ```

   This measures how much of the sample's energy lies in the top-K principal directions, normalized by the expected variance of a clean sample there.

4. **Z-normalizes** `S_K` using trusted-set statistics for each K, then **aggregates** across the K-range `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]` with `max` (default).

5. **Reports** AUC, TPR at 1%/5%/10% FPR, and saves ROC curves, score distributions, and per-K AUC heatmaps.

---

## Repository Structure

```
LoREx/
├── main.py                          Entry point: argument parsing, pipeline orchestration
├── attacks.py                       Attack factory: loads backdoored encoder + trigger model
├── dataset.py                       Dataset factory: builds trusted + test loaders
│
├── lorex/
│   ├── __init__.py                  Public API re-exports
│   ├── scoring.py                   MS-Spectral scoring (spectral_score, ms_spectral, K_RANGE)
│   ├── whitening.py                 ZCA whitening + PCA decomposition (whiten_and_pca)
│   ├── features.py                  Feature extraction from SSL encoders
│   ├── metrics.py                   AUC, TPR@FPR, full_metrics, save_metrics
│   └── viz.py                       ROC curves, score distributions, per-K AUC heatmap
│
├── attacks/
│   ├── BadCLIP/                     BadCLIP attack implementation + trigger patches
│   ├── BadEncoder/                  BadEncoder attack implementation + trigger
│   ├── DRUPE/                       DRUPE attack implementation
│   └── INACTIVE/                    INACTIVE attack implementation + UNet trigger model
│
├── scripts/
│   ├── run_badencoder_cifar10.sh    Run detection on BadEncoder / CIFAR-10
│   ├── run_drupe_cifar10.sh         Run detection on DRUPE / CIFAR-10
│   ├── run_drupe_imagenet.sh        Run detection on DRUPE / ImageNet (CLIP)
│   ├── run_badclip_imagenet.sh      Run detection on BadCLIP / ImageNet
│   ├── run_inactive_clip_cifar10.sh Run detection on INACTIVE / CLIP
│   ├── sweep_n_trusted.sh           Ablation: vary trusted set size
│   └── sweep_poison_rate.sh         Ablation: vary poison fraction
│
└── tmp/
    └── ms_spectral_final.py         Reference scoring implementation
```

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision numpy scipy scikit-learn matplotlib pillow
```

**Data prerequisites** (adjust paths in the run scripts):
- **CIFAR-10**: Download via `torchvision.datasets.CIFAR10` or point `--data_dir` to a local copy.
- **ImageNet validation set**: Set `--imagenet_val_dir` to the ILSVRC2012 validation directory.

---

## Quick Start

Run detection on **BadEncoder / CIFAR-10** (fastest, expected AUC ≈ 1.00):

```bash
.venv/bin/python main.py \
  --attack badencoder \
  --dataset badencoder_cifar10_pair \
  --attack_ckpt_path ./attacks/BadEncoder/output/cifar10/gtsrb_backdoored_encoder/model_200.pth \
  --trigger_path ./attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --badencoder_usage_info cifar10 \
  --data_dir /path/to/cifar10 \
  --device cuda \
  --n_trusted 5000 --n_test 200 \
  --output_dir ./outputs/badencoder_cifar10
```

Output is saved to `./outputs/badencoder_cifar10/`:
- `metrics/results.json` — AUC, TPR@FPR, per-K AUC breakdown
- `figures/roc_curves.png` — ROC curve
- `figures/score_distributions.png` — clean vs poison score histograms
- `figures/per_k_auc_heatmap.png` — detection AUC for each K value

---

## Running All Attack Configurations

Use the pre-configured scripts in `scripts/`. Run each from the project root:

### BadEncoder / CIFAR-10

```bash
bash scripts/run_badencoder_cifar10.sh
```

### DRUPE / CIFAR-10

```bash
bash scripts/run_drupe_cifar10.sh
```

### DRUPE / ImageNet (CLIP encoder)

```bash
bash scripts/run_drupe_imagenet.sh
```

### BadCLIP / ImageNet

```bash
bash scripts/run_badclip_imagenet.sh
```

### INACTIVE / CLIP

```bash
bash scripts/run_inactive_clip_cifar10.sh
```

---

## The `--dataset` and `--trusted_dataset` arguments

These two arguments control separate concerns:

### `--dataset` — test images (attack-specific)

Specifies which images are evaluated and how triggers are applied.

| Choice | Description |
|---|---|
| `badencoder_cifar10_pair` | CIFAR-10 NPZ + BadEncoder trigger (pre-paired) |
| `cifar10_npz_pair` | CIFAR-10 NPZ + generic NPZ trigger (pre-paired) |
| `svhn_npz_pair` | SVHN NPZ + NPZ trigger (pre-paired) |
| `gtsrb_npz_pair` | GTSRB NPZ + NPZ trigger (pre-paired) |
| `stl10_npz_pair` | STL-10 NPZ + NPZ trigger (pre-paired) |
| `imagenet_subset_pair` | ImageNet val + DRUPE-style NPZ trigger (pre-paired) |
| `badclip_imagenet_pair` | ImageNet val + BadCLIP optimized patch trigger (pre-paired) |
| `cifar10_224_clean` | CIFAR-10 224px NPZ, trigger applied on-the-fly via UNet |
| `imagenet_clean` | ImageNet val, trigger applied on-the-fly via UNet |
| `custom_folder_clean` | Custom ImageFolder (`--data_dir`), trigger via UNet |

For `*_pair` datasets, the loader yields `(poisoned_img, clean_img, label, trigger_label)` 4-tuples.
For `*_clean` datasets, the loader yields `(clean_img, label)` 2-tuples and the attack's `trigger_model` generates poisoned variants.

### `--trusted_dataset` — clean images for whitening (independent of attack)

Specifies a completely independent clean dataset used to estimate the whitening matrix.
This dataset must not overlap with the attacker's training data.
If omitted, LoREx falls back to using a clean subset of `--dataset` itself.

| Choice | Source | Format |
|---|---|---|
| `cifar10` | `<trusted_data_dir>/train[_224].npz` | NPZ |
| `svhn` | `<trusted_data_dir>/train[_224].npz` | NPZ |
| `gtsrb` | `<trusted_data_dir>/train[_224].npz` | NPZ |
| `stl10` | `<trusted_data_dir>/train[_224].npz` | NPZ |
| `imagenet` | `<trusted_data_dir>/` ImageFolder tree | ImageFolder |
| `custom_npz` | `<trusted_data_dir>` (exact .npz path) | NPZ |
| `custom_folder` | `<trusted_data_dir>/` ImageFolder tree | ImageFolder |

Add `--trusted_use_224` when using CLIP-based encoders to load the `_224.npz` variant.

**Example — use SVHN as trusted set for a CIFAR-10 attack:**

```bash
.venv/bin/python main.py \
  --attack badencoder --dataset badencoder_cifar10_pair \
  --attack_ckpt_path ... --trigger_path ... \
  --data_dir /data/cifar10 \
  --trusted_dataset svhn --trusted_data_dir /data/svhn \
  --n_trusted 5000 --device cuda
```

**Example — use STL-10 (224px) as trusted set for INACTIVE CLIP:**

```bash
.venv/bin/python main.py \
  --attack inactive --dataset imagenet_clean \
  --attack_ckpt_path ... --unet_path ... \
  --imagenet_val_dir /data/imagenet/val --labels_csv ... \
  --trusted_dataset stl10 --trusted_data_dir /data/stl10 --trusted_use_224 \
  --n_trusted 5000 --device cuda
```

**Example — use a custom ImageFolder as trusted set:**

```bash
.venv/bin/python main.py \
  --attack badclip --dataset badclip_imagenet_pair \
  --attack_ckpt_path ... --imagenet_val_dir ... --labels_csv ... \
  --trusted_dataset custom_folder --trusted_data_dir /path/to/my/clean/dataset \
  --n_trusted 5000 --device cuda
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--attack` | required | Attack type: `badencoder`, `badclip`, `drupe`, `inactive` |
| `--dataset` | required | Test dataset configuration (see table above) |
| `--trusted_dataset` | `None` (use `--dataset`) | Clean trusted source for whitening (see table above) |
| `--trusted_data_dir` | `<data_dir>/<trusted_dataset>` | Path to trusted dataset |
| `--trusted_use_224` | `False` | Load 224px NPZ for CLIP-based encoders |
| `--trusted_split` | `train` | NPZ split for trusted set (`train` or `test`) |
| `--attack_ckpt_path` | `""` | Path to backdoored encoder checkpoint |
| `--n_trusted` | `5000` | Number of trusted clean samples for whitening |
| `--n_test` | `200` | Number of clean test samples |
| `--n_poison` | same as `--n_test` | Number of poison test samples |
| `--agg` | `max` | Aggregation across K values: `max`, `mean`, `median`, `p90`, `top3mean` |
| `--k_range` | `[1,2,3,4,6,8,12,16,24,32,48,64]` | K values for MS-Spectral |
| `--device` | `cuda` | Compute device |
| `--output_dir` | `./outputs` | Directory for results and figures |

---

## Output Format

`metrics/results.json` contains:

```json
{
  "attack": "badencoder",
  "dataset": "npz_pair",
  "auc": 1.0,
  "tpr@1%": 1.0,
  "tpr@5%": 1.0,
  "tpr@10%": 1.0,
  "per_k_aucs": {"1": 0.95, "2": 0.98, "4": 1.0, ...},
  "n_trusted": 5000,
  "n_test_clean": 200,
  "n_test_poison": 200,
  "k_range": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
  "agg": "max"
}
```
