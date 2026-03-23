#!/usr/bin/env bash
set -euo pipefail

python ./lorex/main.py \
  --attack badclip \
  --dataset imagenet_subset_pair \
  --attack_ckpt_path ./BadCLIP/badclip/checkpoints/clip_backdoor.pth \
  --trigger_path ./BadCLIP/trigger/trigger_pt_white_185_24.npz \
  --imagenet_val_dir ./BadCLIP/data/ImageNet1K/validation \
  --labels_csv ./BadCLIP/data/ImageNet1K/validation/labels.csv \
  --classes_py ./BadCLIP/data/ImageNet1K/validation/classes.py \
  --clean_subset_frac 0.1 \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --trigger_chance 0.01 \
  --threshold_percentile 99