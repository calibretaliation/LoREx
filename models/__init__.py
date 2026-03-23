"""SSL model registry for LoREx.

Provides a lightweight registry that maps model-type names to their feature
extraction functions.  Adding support for a new SSL model requires only:

    1. Create models/<new_model>.py
    2. Call register_model("<name>") on a class that implements feature_fn()
    3. Import it below so the registry is populated at import time

Currently registered:
    "simclr" — ResNet-based contrastive encoder  (feature = model.f(img))
    "clip"   — OpenAI CLIP visual encoder         (feature = model.visual(img))

Example — plug in a DINO encoder:

    # models/dino.py
    from models import register_model

    @register_model("dino")
    class DINOEncoder:
        @staticmethod
        def feature_fn(model, img):
            return model(img)

    # Then in attacks.py:
    from models import get_feature_fn
    feat_fn = get_feature_fn("dino")
"""

from typing import Callable, Type, Dict

import torch

_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Class decorator that registers an SSL encoder type by name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_feature_fn(name: str) -> Callable:
    """Return the feature_fn for the registered model type *name*.

    Args:
        name: registered model type (e.g. "simclr", "clip").

    Returns:
        Callable(model, img: Tensor) -> Tensor
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown SSL model type: {name!r}. Available: {available}")
    return _REGISTRY[name].feature_fn


# Populate registry by importing all model modules
from . import simclr  # noqa: E402, F401
from . import clip    # noqa: E402, F401
