#!/usr/bin/env bash
set -euo pipefail

python ./LoREx/main.py \
  --attack drupe \
  --dataset cifar10_npz_pair \
  --attack_ckpt_path ./DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_gtsrb_t12/best.pth \
  --trigger_path ./DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --data_dir ./data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/drupe_cifar10
