#!/usr/bin/env bash
set -euo pipefail

python /media/lambda/SSD1/nhat/lorex/main.py \
  --attack inactive \
  --dataset cifar10_224_clean \
  --inactive_model_type clip \
  --attack_ckpt_path /media/lambda/SSD1/nhat/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth \
  --unet_path /media/lambda/SSD1/nhat/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt \
  --data_dir /media/lambda/SSD1/nhat/DRUPE/data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --trigger_batch_num 10 \
  --threshold_percentile 99 \
  --plot_heatmaps