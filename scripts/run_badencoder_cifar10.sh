#!/usr/bin/env bash
set -euo pipefail

python ./LoREx/main.py \
  --attack badencoder \
  --dataset badencoder_cifar10_pair \
  --attack_ckpt_path ./BadEncoder/output/cifar10/stl10_backdoored_encoder/model_200.pth \
  --badencoder_usage_info cifar10 \
  --trigger_path ./BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --data_dir ./data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/badencoder_cifar10
