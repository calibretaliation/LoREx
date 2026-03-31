import sys, os, torch, numpy as np
from experiment_lorex_aug_comprehensive import load_attack, SimCLRTestDataset, extract_test_features, score_lorex_aug, precompute_lorex_aug_stats, ATTACK_CONFIGS, build_trusted_loader, BATCH_SIZE, NUM_WORKERS, SEED, DATA_ROOT, extract_features

device = torch.device('cuda')

cfg = next(c for c in ATTACK_CONFIGS if c['name'] == 'BadEncoder-SimCLR')
attack = load_attack(cfg, device)

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

rng = np.random.default_rng(SEED)
cifar_x = np.load(f"{DATA_ROOT}/cifar10/train.npz")["x"]
n_total = 1000
n_poison = 100
indices = rng.choice(len(cifar_x), size=n_total, replace=False)
images = cifar_x[indices]
poison_flags = np.zeros(n_total, dtype=bool)
poison_flags[rng.choice(n_total, size=n_poison, replace=False)] = True

dataset = SimCLRTestDataset(
    images_np=images,
    poison_flags=poison_flags,
    trigger_path=cfg["trigger_path"],
    trigger_size=cfg["trigger_size"],
    transform=attack.transform,
)
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
test_feats, test_labels = extract_test_features(loader, attack.model, device, attack.feature_fn)
scores = score_lorex_aug(test_feats, stats)
from lorex.metrics import full_metrics
m = full_metrics(scores[test_labels==0], scores[test_labels==1])
print("BadEncoder-SimCLR AUC on train.npz:", m['auc'])
