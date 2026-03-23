#!/usr/bin/env bash
# Run from LoREx/ project root
set -euo pipefail

.venv/bin/python main.py \
  --attack badclip \
  --dataset badclip_imagenet_pair \
  --attack_ckpt_path ./attacks/BadCLIP/badclip/logs/nodefence_ours_final/checkpoints/epoch_10.pt \
  --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
  --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
  --patch_type ours_tnature \
  --patch_location middle \
  --patch_size 16 \
  --patch_name './attacks/BadCLIP/opti_patches/tnature_eda_aug_bs64_ep50_16_middle_01_05_pos_neg_tri*500.jpg' \
  --clean_subset_frac 0.1 \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/badclip_imagenet
