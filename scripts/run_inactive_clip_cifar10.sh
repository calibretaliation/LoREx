#!/usr/bin/env bash
# Run from LoREx/ project root
# INACTIVE CLIP encoder (backdoored on CIFAR-10), evaluated on ImageNet trusted set
set -euo pipefail

.venv/bin/python main.py \
  --attack inactive \
  --dataset imagenet_clean \
  --inactive_model_type clip \
  --attack_ckpt_path ./attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth \
  --unet_path ./attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt \
  --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
  --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
  --trusted_frac 0.1 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/inactive_clip_cifar10
