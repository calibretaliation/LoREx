#!/usr/bin/env bash
# Run from LoREx/ project root
# DRUPE with CLIP encoder on ImageNet
set -euo pipefail

.venv/bin/python main.py \
  --attack badclip \
  --dataset imagenet_subset_pair \
  --attack_ckpt_path ./attacks/DRUPE/output/CLIP/backdoor/truck/model_200.pth \
  --trigger_path ./attacks/BadCLIP/trigger/trigger_pt_white_185_24.npz \
  --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
  --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
  --clean_subset_frac 0.1 \
  --device cuda \
  --batch_size 128 \
  --num_workers 4 \
  --n_trusted 5000 \
  --n_test 200 \
  --output_dir ./outputs/drupe_imagenet
