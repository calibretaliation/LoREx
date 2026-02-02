#!/usr/bin/env bash
set -euo pipefail

python /media/lambda/SSD1/nhat/lorex/main.py \
  --attack badencoder \
  --dataset badencoder_cifar10_pair \
  --attack_ckpt_path /media/lambda/SSD1/nhat/BadEncoder/output/cifar10/stl10_backdoored_encoder/model_200.pth \
  --badencoder_usage_info cifar10 \
  --trigger_path /media/lambda/SSD1/nhat/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --data_dir /media/lambda/SSD1/nhat/data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --trigger_chance 0.01 \
  --threshold_percentile 99