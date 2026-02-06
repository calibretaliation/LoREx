#!/usr/bin/env bash
set -euo pipefail

python ./lorex/main.py \
  --attack inactive \
  --dataset cifar10_224_clean \
  --inactive_model_type clip \
  --attack_ckpt_path ./INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth \
  --unet_path ./INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt \
  --data_dir ./DRUPE/data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --trigger_batch_num 10 \
  --threshold_percentile 99 \
  --plot_heatmaps