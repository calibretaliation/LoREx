#!/usr/bin/env bash
set -euo pipefail

python /media/lambda/SSD1/nhat/lorex/main.py \
  --attack badclip \
  --dataset imagenet_subset_pair \
  --attack_ckpt_path /media/lambda/SSD1/nhat/BadCLIP/badclip/checkpoints/clip_backdoor.pth \
  --trigger_path /media/lambda/SSD1/nhat/BadCLIP/trigger/trigger_pt_white_185_24.npz \
  --imagenet_val_dir /media/lambda/SSD1/nhat/BadCLIP/data/ImageNet1K/validation \
  --labels_csv /media/lambda/SSD1/nhat/BadCLIP/data/ImageNet1K/validation/labels.csv \
  --classes_py /media/lambda/SSD1/nhat/BadCLIP/data/ImageNet1K/validation/classes.py \
  --clean_subset_frac 0.1 \
  --device cuda \
  --batch_size 64 \
  --num_workers 4 \
  --trigger_chance 0.01 \
  --threshold_percentile 99