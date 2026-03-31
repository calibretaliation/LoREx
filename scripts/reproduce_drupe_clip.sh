#!/usr/bin/env bash
# reproduce_drupe_clip.sh — Retrain the DRUPE attack on CLIP-RN50
#
# Usage:  cd attacks/DRUPE && bash ../../scripts/reproduce_drupe_clip.sh
#
# Prerequisite: the DRUPE code expects ImageNet data symlinked at ./data/imagenet/
# and the reference file at ./reference/CLIP/truck.npz (already present).
#
# For CLIP mode, DRUPE loads the pretrained CLIP-RN50 from pkgs/openai/clip
# directly — no external clean encoder weights are needed.
#
# Output checkpoint:
#   ./DRUPE_results/drupe/pretrain_CLIP_sf0.01/downstream_cifar10_t9/best.pth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRUPE_DIR="${SCRIPT_DIR}/../attacks/DRUPE"
cd "${DRUPE_DIR}"

# Ensure ImageNet shadow data is symlinked
if [ ! -d "./data/imagenet" ]; then
    echo "Symlinking ImageNet data..."
    mkdir -p ./data
    ln -sfn /media/lambda/SSD1/nhat/data/imagenet ./data/imagenet
fi

echo "=== DRUPE CLIP-RN50 training (target: truck / cifar10 label 9) ==="
echo "Working directory: $(pwd)"
echo "Output: ./DRUPE_results/drupe/pretrain_CLIP_sf0.01/downstream_cifar10_t9/"

python -u main.py \
  --mode drupe \
  --batch_size 16 \
  --shadow_dataset imagenet \
  --encoder_usage_info CLIP \
  --downstream_dataset cifar10 \
  --target_label 9 \
  --shadow_fraction 0.01 \
  --gpu 0 \
  --trigger_file ./trigger/trigger_pt_white_185_24.npz \
  --lr 0.001 --epochs 200 \
  --reference_file ./reference/CLIP/truck.npz

echo ""
echo "=== Training complete ==="
CKPT="./DRUPE_results/drupe/pretrain_CLIP_sf0.01/downstream_cifar10_t9/best.pth"
if [ -f "${CKPT}" ]; then
    echo "Checkpoint saved: ${CKPT}"
else
    echo "WARNING: best.pth not found — check training logs"
fi
