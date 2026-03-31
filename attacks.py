"""Attack model builders for LoREx.

Each builder loads a backdoored SSL encoder checkpoint and returns an
AttackSpec that bundles the model with its feature extraction function.

Feature functions are defined in models/ and registered by SSL model type
("simclr", "clip"), so adding a new encoder type requires only adding a new
file in models/.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import torch

from models import get_feature_fn


@dataclass
class AttackSpec:
    """Everything LoREx needs to evaluate a backdoored encoder."""
    name: str
    model: torch.nn.Module
    feature_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    processor: Optional[object] = None       # e.g. CLIP image preprocessor
    trigger_model: Optional[torch.nn.Module] = None  # e.g. INACTIVE UNet
    transform: Optional[Callable] = None     # standard input transform (PIL → Tensor)


class ConfigTxt:
    """Simple key=value config file parser (used by INACTIVE / DeDe)."""

    def __init__(self, path: str):
        self.path = path
        self.params = self._load_config()

    def _load_config(self):
        params = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                key, value = line.strip().split("=")
                value = value.strip()
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                    value = float(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                params[key.strip()] = value
        return params

    def __getattr__(self, key):
        return self.params.get(key, None)

    def __setattr__(self, key, value):
        if key in {"path", "params", "_load_config"}:
            super().__setattr__(key, value)
        else:
            self.params[key] = value


def build_drupe_attack(ckpt_path: str, device: torch.device) -> AttackSpec:
    from DRUPE.models import SimCLR
    from BadEncoder.datasets.cifar10_dataset import test_transform_cifar10

    model = SimCLR()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return AttackSpec(
        name="drupe", model=model,
        feature_fn=get_feature_fn("simclr"),
        transform=test_transform_cifar10,
    )


def build_badencoder_attack(ckpt_path: str, device: torch.device, usage_info: str) -> AttackSpec:
    from BadEncoder.models import get_encoder_architecture_usage
    from BadEncoder.datasets.cifar10_dataset import test_transform_cifar10

    args = SimpleNamespace(encoder_usage_info=usage_info)
    model = get_encoder_architecture_usage(args)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return AttackSpec(
        name="badencoder", model=model,
        feature_fn=get_feature_fn("simclr"),
        transform=test_transform_cifar10,
    )


def build_badclip_attack(ckpt_path: str, device: torch.device, clip_name: str = "RN50") -> AttackSpec:
    import sys, pathlib
    _attacks_dir = str(pathlib.Path(__file__).resolve().parent / "attacks")
    if _attacks_dir not in sys.path:
        sys.path.insert(0, _attacks_dir)
    from BadCLIP.pkgs.openai.clip import load as load_clip

    model, processor = load_clip(name=clip_name, pretrained=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return AttackSpec(
        name="badclip", model=model,
        feature_fn=get_feature_fn("clip"), processor=processor,
        transform=processor.process_image,
    )


def build_drupe_clip_attack(ckpt_path: str, device: torch.device, clip_name: str = "RN50") -> AttackSpec:
    """Load a DRUPE-backdoored CLIP model.

    DRUPE trains only the visual encoder but saves the full CLIP state_dict
    (all 489 keys). We load pretrained CLIP-RN50 and overwrite with the
    checkpoint via strict=False.
    """
    spec = build_badclip_attack(ckpt_path, device, clip_name=clip_name)
    spec.name = "drupe_clip"
    return spec


def build_inactive_attack(
    device: torch.device,
    model_type: str,
    clip_ckpt_path: Optional[str],
    unet_path: Optional[str],
    config_path: Optional[str],
    encoder_path: Optional[str],
) -> AttackSpec:
    model_type = (model_type or "clip").lower().strip()

    def _load_unet():
        if not unet_path:
            return None
        from INACTIVE.optimize_filter.tiny_network import U_Net_tiny
        unet = U_Net_tiny(img_ch=3, output_ch=3)
        unet_sd = torch.load(unet_path, map_location=device, weights_only=False)
        unet.load_state_dict(unet_sd["model_state_dict"])
        return unet.to(device).eval()

    if model_type == "clip":
        if not clip_ckpt_path:
            raise ValueError("inactive clip requires --attack_ckpt_path")
        import sys, pathlib
        _attacks_dir = str(pathlib.Path(__file__).resolve().parent / "attacks")
        if _attacks_dir not in sys.path:
            sys.path.insert(0, _attacks_dir)
        from BadCLIP.pkgs.openai.clip import load as load_clip

        model, processor = load_clip(name="RN50", pretrained=True)
        ckpt = torch.load(clip_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        return AttackSpec(
            name="inactive", model=model,
            feature_fn=get_feature_fn("clip"), processor=processor,
            trigger_model=_load_unet(),
            transform=processor.process_image,
        )

    if model_type == "simclr":
        if not config_path:
            raise ValueError("inactive simclr requires --inactive_config_path")
        from DeDe.new_defense.utils import get_backdoor_encoder

        cfg = ConfigTxt(config_path)
        if encoder_path:
            cfg.encoder_path = encoder_path
        cfg.DEVICE = str(device)
        model = get_backdoor_encoder(cfg).to(device).eval()
        from BadEncoder.datasets.cifar10_dataset import test_transform_cifar10
        return AttackSpec(
            name="inactive", model=model,
            feature_fn=get_feature_fn("simclr"),
            trigger_model=_load_unet(),
            transform=test_transform_cifar10,
        )

    raise ValueError(f"Unsupported inactive model_type: {model_type!r}")


def build_attack_from_args(args) -> AttackSpec:
    device = torch.device(args.device)
    ckpt_path = args.attack_ckpt_path or args.ckpt_path
    if args.attack == "drupe":
        if not ckpt_path:
            raise ValueError("drupe requires --attack_ckpt_path")
        return build_drupe_attack(ckpt_path, device)
    if args.attack == "badencoder":
        if not ckpt_path:
            raise ValueError("badencoder requires --attack_ckpt_path")
        return build_badencoder_attack(ckpt_path, device, args.badencoder_usage_info)
    if args.attack == "badclip":
        if not ckpt_path:
            raise ValueError("badclip requires --attack_ckpt_path")
        return build_badclip_attack(ckpt_path, device, clip_name=args.clip_name)
    if args.attack == "inactive":
        return build_inactive_attack(
            device=device,
            model_type=args.inactive_model_type,
            clip_ckpt_path=ckpt_path,
            unet_path=args.unet_path,
            config_path=args.inactive_config_path,
            encoder_path=args.inactive_encoder_path,
        )
    raise ValueError(f"Unsupported attack: {args.attack!r}")

def build_drupe_imagenet_attack(ckpt_path: str, device: torch.device) -> AttackSpec:
    from DRUPE.models.imagenet_model import ImageNetResNet
    from BadEncoder.datasets.imagenet_dataset import test_transform_imagenet

    model = ImageNetResNet()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.visual.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    def feature_fn(m, x):
        return m.visual(x)

    return AttackSpec(
        name="drupe_imagenet", model=model,
        feature_fn=feature_fn,
        transform=test_transform_imagenet,
    )

def build_drupe_imagenet_attack(ckpt_path: str, device: torch.device) -> AttackSpec:
    from DRUPE.models.imagenet_model import ImageNetResNet
    from torchvision import transforms

    test_transform_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = ImageNetResNet()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.visual.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    def feature_fn(m, x):
        return m.visual(x)

    return AttackSpec(
        name="drupe_imagenet", model=model,
        feature_fn=feature_fn,
        transform=test_transform_imagenet,
    )
