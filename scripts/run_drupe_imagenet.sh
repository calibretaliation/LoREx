#!/usr/bin/env bash
set -euo pipefail

# DRUPE with CLIP encoder on ImageNet
python ./LoREx/main.py \
  --attack badclip \
  --dataset imagenet_subset_pair \
  --attack_ckpt_path ./DRUPE/DRUPE_results/drupe/CLIP_100/downstream_cifar10_t0/epoch100.pth \
  --trigger_path ./DRUPE/trigger/trigger_pt_white_185_24.npz \
  --imagenet_val_dir ./BadCLIP/data/ImageNet1K/validation \
  --labels_csv ./BadCLIP/data/ImageNet1K/validation/labels.csv \
  --classes_py ./BadCLIP/data/ImageNet1K/validation/classes.py \
  --clean_subset_frac 0.1 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/drupe_imagenet
