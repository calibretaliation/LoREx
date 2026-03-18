"""CLIP visual encoder interface for LoREx.

Registered as "clip".  Feature extraction uses the visual branch of a CLIP
model via model.visual(img).

Compatible attacks: BadCLIP, INACTIVE (clip variant).
"""

import torch
from models import register_model


@register_model("clip")
class CLIPEncoder:
    """Feature extraction interface for OpenAI CLIP visual encoders.

    These models expose a .visual() method that maps images to visual
    embeddings in the shared image-text embedding space.
    """

    @staticmethod
    def feature_fn(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        """Extract visual features from a CLIP model.

        Args:
            model: CLIP model with a .visual(img) method.
            img: preprocessed image batch, shape (B, C, H, W).

        Returns:
            Visual embedding tensor, shape (B, d).
        """
        return model.visual(img)
