"""SimCLR SSL encoder interface for LoREx.

Registered as "simclr".  Feature extraction uses the projection head's
penultimate representation via model.f(img).

Compatible attacks: DRUPE, BadEncoder, INACTIVE (simclr variant).
"""

import torch
from models import register_model


@register_model("simclr")
class SimCLREncoder:
    """Feature extraction interface for SimCLR-style encoders.

    These encoders expose a .f() method that maps images to embeddings
    (before the projection head).
    """

    @staticmethod
    def feature_fn(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        """Extract features from a SimCLR encoder.

        Args:
            model: SimCLR encoder with a .f(img) method.
            img: image batch, shape (B, C, H, W).

        Returns:
            Embedding tensor, shape (B, d).
        """
        return model.f(img)
