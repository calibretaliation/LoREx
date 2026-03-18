#!/usr/bin/env bash
set -euo pipefail

# CLIP encoder pretrained on ImageNet; use ImageNet as trusted clean set
python ./LoREx/main.py \
  --attack inactive \
  --dataset imagenet_clean \
  --inactive_model_type clip \
  --attack_ckpt_path ./INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth \
  --unet_path ./INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt \
  --imagenet_val_dir ./BadCLIP/data/ImageNet1K/validation \
  --labels_csv ./BadCLIP/data/ImageNet1K/validation/labels.csv \
  --trusted_frac 0.1 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/inactive_clip_cifar10
