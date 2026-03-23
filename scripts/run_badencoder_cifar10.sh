#!/usr/bin/env bash
# Run from LoREx/ project root
set -euo pipefail

.venv/bin/python main.py \
  --attack badencoder \
  --dataset badencoder_cifar10_pair \
  --attack_ckpt_path ./attacks/BadEncoder/output/cifar10/gtsrb_backdoored_encoder/model_200.pth \
  --badencoder_usage_info cifar10 \
  --trigger_path ./attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --data_dir /media/lambda/SSD1/nhat/data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/badencoder_cifar10
