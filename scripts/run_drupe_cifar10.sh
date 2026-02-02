#!/usr/bin/env bash
set -euo pipefail

python /media/lambda/SSD1/nhat/lorex/main.py \
  --attack drupe \
  --dataset cifar10_npz_pair \
  --attack_ckpt_path /media/lambda/SSD1/nhat/DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_gtsrb_t12/best.pth \
  --trigger_path /media/lambda/SSD1/nhat/DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --data_dir /media/lambda/SSD1/nhat/data/cifar10 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --trigger_chance 0.01 \
  --threshold_percentile 99 \
  --plot_heatmaps