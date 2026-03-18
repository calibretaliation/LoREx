# Adaptive Attack (BadEncoder + Whitening Evasion)

This folder implements an adaptive BadEncoder-style backdoor that learns a dynamic trigger through a UNet generator while steering the trigger signal away from whitening-sensitive subspaces. The original `BadEncoder` code is left untouched and imported directly.

## Key ideas
- UNet generator writes a subtle perturbation onto the normalized input and is trained jointly with the student encoder.
- Loss terms: clean consistency (teacher vs. student), backdoor target alignment, whitening evasion (project trigger onto low-variance eigenvectors and suppress), and pixel-level stealth.
- Low-variance subspace is estimated from the *evasion dataset* using the frozen teacher encoder (should match the defense distribution).
- Backdoor objective can be trained directly on the downstream distribution via `--target_dataset`.

## Quick start
1. Ensure the BadEncoder data layout is available (e.g., `./data/cifar10/train.npz`, `test.npz`).
2. Prepare `reference_file` and `trigger_file` as expected by the original BadEncoder loaders.
3. Run training:

```bash
python adaptive_attack/train.py \
  --shadow_dataset cifar10 \
  --target_dataset gtsrb \
  --evasion_dataset gtsrb \
  --reference_file /media/lambda/SSD1/nhat/BadEncoder/reference/cifar10/priority.npz \
  --reference_label 12 \
  --trigger_file /media/lambda/SSD1/nhat/BadEncoder/trigger/trigger_pt_white_21_10_ap_replace.npz \
  --teacher_ckpt /media/lambda/SSD1/nhat/BadEncoder/output/cifar10/clean_encoder/model_1000.pth \
  --student_init_ckpt /media/lambda/SSD1/nhat/BadEncoder/output/cifar10/clean_encoder/model_1000.pth \
  --use_residual_target \
  --results_dir ./adaptive_attack/adaptive_attack_runs \
  --encoder_usage_info cifar10 \
  --epochs 200 --batch_size 128
```

The script saves checkpoints (student encoder + generator + whitening subspace) to the chosen results directory.

## Main arguments
- `--lambda_clean`, `--lambda_target`, `--lambda_evasion`, `--lambda_stealth`: weights for the four losses.
- `--eig_k`: number of low-variance eigenvectors to suppress.
- `--max_eigen_batches`: cap batches used for eigen estimation (set to `-1` to use the full shadow set).
- `--teacher_ckpt`: path to a truly clean teacher encoder checkpoint (frozen).
- `--student_init_ckpt`: initialization checkpoint for the student (defaults to `--teacher_ckpt`).
- `--pretrained_encoder`: deprecated alias for setting both teacher and student init.
- `--encoder_usage_info`: supports `cifar10`, `stl10`, `imagenet`, `CLIP` following BadEncoder conventions.
- `--target_dataset`: dataset used to train the backdoor objective (recommended: downstream dataset you evaluate ASR on).
- `--evasion_dataset`: dataset used to compute whitening-evasion subspace (recommended: same distribution the defense monitors).
- `--use_residual_target`: enables a residual target objective intended to be more discriminative for downstream ASR.

## Notes
- Evaluation with the original KNN monitor is optional and runs if the downstream loaders are available for the chosen dataset.
- The generator clamps perturbations in normalized space to keep outputs bounded while training.