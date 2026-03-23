#!/usr/bin/env bash
# Experiment 2: Sweep poison rate (n_poison) across all attacks.
# Uses the best detector from Experiment 1 (default: mahal).
# Run from LoREx/ project root.
set -euo pipefail

PYTHON=.venv/bin/python
DETECTOR="${DETECTOR:-mahal}"
POISON_COUNTS=(1 2 5 10 20 50 100 200)

run_attack() {
  local attack_name="$1"
  local n_poison="$2"
  local outdir="./outputs/sweep_poison_rate/${attack_name}_np${n_poison}"

  echo "── ${attack_name}  n_poison=${n_poison}  detector=${DETECTOR} ──"

  case "$attack_name" in
    badencoder_cifar10)
      $PYTHON main.py \
        --attack badencoder \
        --dataset badencoder_cifar10_pair \
        --attack_ckpt_path ./attacks/BadEncoder/output/cifar10/gtsrb_backdoored_encoder/model_200.pth \
        --badencoder_usage_info cifar10 \
        --trigger_path ./attacks/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
        --data_dir /media/lambda/SSD1/nhat/data/cifar10 \
        --device cuda --batch_size 128 --num_workers 4 \
        --n_trusted 5000 --n_test 200 --n_poison "$n_poison" \
        --detector "$DETECTOR" \
        --output_dir "$outdir"
      ;;
    drupe_cifar10)
      $PYTHON main.py \
        --attack drupe \
        --dataset cifar10_npz_pair \
        --attack_ckpt_path ./attacks/DRUPE/output/cifar10/gtsrb_backdoored_encoder/model_200.pth \
        --trigger_path ./attacks/DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz \
        --data_dir /media/lambda/SSD1/nhat/data/cifar10 \
        --device cuda --batch_size 128 --num_workers 4 \
        --n_trusted 5000 --n_test 200 --n_poison "$n_poison" \
        --detector "$DETECTOR" \
        --output_dir "$outdir"
      ;;
    drupe_imagenet)
      $PYTHON main.py \
        --attack badclip \
        --dataset imagenet_subset_pair \
        --attack_ckpt_path ./attacks/DRUPE/output/CLIP/backdoor/truck/model_200.pth \
        --trigger_path ./attacks/BadCLIP/trigger/trigger_pt_white_185_24.npz \
        --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
        --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
        --clean_subset_frac 0.1 \
        --device cuda --batch_size 128 --num_workers 4 \
        --n_trusted 5000 --n_test 200 --n_poison "$n_poison" \
        --detector "$DETECTOR" \
        --output_dir "$outdir"
      ;;
    inactive_clip)
      $PYTHON main.py \
        --attack inactive \
        --dataset imagenet_clean \
        --inactive_model_type clip \
        --attack_ckpt_path ./attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/model_200.pth \
        --unet_path ./attacks/INACTIVE/output/CLIP/cifar10_backdoored_encoder/2025-12-03-08:46:43/unet_filter_200_trained.pt \
        --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
        --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
        --trusted_frac 0.1 \
        --device cuda --batch_size 128 --num_workers 4 \
        --n_trusted 5000 --n_test 200 --n_poison "$n_poison" \
        --detector "$DETECTOR" \
        --output_dir "$outdir"
      ;;
    badclip_imagenet)
      $PYTHON main.py \
        --attack badclip \
        --dataset badclip_imagenet_pair \
        --attack_ckpt_path ./attacks/BadCLIP/badclip/logs/nodefence_ours_final/checkpoints/epoch_10.pt \
        --imagenet_val_dir /media/lambda/SSD1/nhat/data/imagenet/ILSVRC2012_DET_val \
        --labels_csv ./attacks/BadCLIP/data/ImageNet1K/validation/labels.csv \
        --patch_type ours_tnature --patch_location middle --patch_size 16 \
        --patch_name './attacks/BadCLIP/opti_patches/tnature_eda_aug_bs64_ep50_16_middle_01_05_pos_neg_tri*500.jpg' \
        --clean_subset_frac 0.1 \
        --device cuda --batch_size 64 --num_workers 4 \
        --n_trusted 5000 --n_test 200 --n_poison "$n_poison" \
        --detector "$DETECTOR" \
        --output_dir "$outdir"
      ;;
  esac
}

ATTACKS=(badencoder_cifar10 drupe_cifar10 drupe_imagenet inactive_clip badclip_imagenet)

for attack in "${ATTACKS[@]}"; do
  for np in "${POISON_COUNTS[@]}"; do
    run_attack "$attack" "$np"
  done
done

echo ""
echo "Poison rate sweep complete. Results in ./outputs/sweep_poison_rate/"
echo "Override detector with: DETECTOR=ocsvm bash scripts/sweep_poison_rate.sh"
