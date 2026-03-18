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

    model = SimCLR()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return AttackSpec(name="drupe", model=model, feature_fn=get_feature_fn("simclr"))


def build_badencoder_attack(ckpt_path: str, device: torch.device, usage_info: str) -> AttackSpec:
    from BadEncoder.models import get_encoder_architecture_usage

    args = SimpleNamespace(encoder_usage_info=usage_info)
    model = get_encoder_architecture_usage(args)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return AttackSpec(name="badencoder", model=model, feature_fn=get_feature_fn("simclr"))


def build_badclip_attack(ckpt_path: str, device: torch.device, clip_name: str = "RN50") -> AttackSpec:
    from BadCLIP.pkgs.openai.clip import load as load_clip

    model, processor = load_clip(name=clip_name, pretrained=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return AttackSpec(
        name="badclip", model=model,
        feature_fn=get_feature_fn("clip"), processor=processor,
    )


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
        return AttackSpec(
            name="inactive", model=model,
            feature_fn=get_feature_fn("simclr"),
            trigger_model=_load_unet(),
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
