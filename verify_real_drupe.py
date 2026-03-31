import torch
import sys
from pathlib import Path

# Add attacks to sys.path
REPO_ROOT = Path("/media/lambda/SSD1/nhat/LoREx")
sys.path.insert(0, str(REPO_ROOT / "attacks"))

from attacks import build_drupe_attack, build_drupe_clip_attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load real DRUPE-SimCLR
try:
    print("Loading real DRUPE-SimCLR...")
    simclr = build_drupe_attack(
        ckpt_path="attacks/DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_gtsrb_t12/best.pth",
        device=device
    )
    print("Successfully loaded real DRUPE-SimCLR.")
except Exception as e:
    print(f"Error loading DRUPE-SimCLR: {e}")

# Load real DRUPE-CLIP
try:
    print("\nLoading real DRUPE-CLIP...")
    clip = build_drupe_clip_attack(
        ckpt_path="attacks/DRUPE/DRUPE_results/drupe/pretrain_CLIP_sf0.2/downstream_cifar10_t0/best.pth",
        device=device
    )
    print("Successfully loaded real DRUPE-CLIP.")
except Exception as e:
    print(f"Error loading DRUPE-CLIP: {e}")

