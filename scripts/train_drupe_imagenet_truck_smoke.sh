#!/usr/bin/env bash
# Quick smoke run for DRUPE CLIP truck setup.
# Run from LoREx/ directory.
set -euo pipefail

cd attacks/DRUPE

mkdir -p data
if [ ! -L "data/imagenet" ] && [ ! -d "data/imagenet" ]; then
    ln -sf /media/lambda/SSD1/nhat/data/imagenet data/imagenet
fi

../../.venv/bin/python -u main.py \
  --mode drupe \
  --batch_size 16 \
  --shadow_dataset imagenet \
  --shadow_fraction 0.0002 \
  --encoder_usage_info CLIP \
  --downstream_dataset cifar10 \
  --target_label 9 \
  --gpu 0 \
  --trigger_file trigger/trigger_pt_white_185_24.npz \
  --lr 0.05 --epochs 2 \
  --reference_file reference/CLIP/truck.npz \
  --nn_epochs 2 \
  --batch_size_downstream 256 \
  --sanity_checks
