import sys, os, torch
from experiment_lorex_aug_comprehensive import load_attack, build_test_dataset, extract_test_features, score_lorex_aug, precompute_lorex_aug_stats, ATTACK_CONFIGS, build_trusted_loader, BATCH_SIZE, NUM_WORKERS, SEED, DATA_ROOT, IMAGENET_TRAIN, DEFAULT_TRUSTED_SIMCLR, extract_features

device = torch.device('cuda')

# Test BadEncoder-SimCLR
cfg = next(c for c in ATTACK_CONFIGS if c['name'] == 'BadEncoder-SimCLR')
attack = load_attack(cfg, device)
# trusted 
trusted_dataset = "cifar10"
trusted_loader = build_trusted_loader(
    trusted_dataset=trusted_dataset,
    data_dir=f"{DATA_ROOT}/{trusted_dataset}",
    transform=attack.transform,
    n_trusted=2000,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_224=False,
    split="train",
    seed=SEED,
)
trusted_feats = extract_features(trusted_loader, attack.model, device, attack.feature_fn)
stats = precompute_lorex_aug_stats(trusted_feats)

test_loader, labels, needs_unet = build_test_dataset(cfg, attack, 1000, 100, SEED)
from experiment_lorex_aug_comprehensive import extract_test_features_unet
if needs_unet:
    test_feats, test_labels = extract_test_features_unet(test_loader, attack.model, attack.trigger_model, device, attack.feature_fn)
else:
    test_feats, test_labels = extract_test_features(test_loader, attack.model, device, attack.feature_fn)

scores = score_lorex_aug(test_feats, stats)
import numpy as np
from lorex.metrics import full_metrics
m = full_metrics(scores[labels==0], scores[labels==1])
print("BadEncoder-SimCLR AUC:", m['auc'])
