import torch
import torch.nn as nn
import random

from .mapping_network import MappingNetwork
from .synthesis_network import SynthesisNetwork


class Generator(nn.Module):
    """
    Updated StyleGAN Generator Wrapper
    Improvements:
    - Lower style mixing probability
    - Dynamic num_ws support
    - Cleaner style mixing logic
    - Safer forward pass
    """

    def __init__(self, z_dim=256, w_dim=256):
        super().__init__()

        self.mapping = MappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            depth=8
        )

        self.synthesis = SynthesisNetwork(w_dim=w_dim)

        # If synthesis defines num_ws use it, else fallback to 10
        self.num_ws = getattr(self.synthesis, "num_ws", 10)

    def make_style_bundle(self, w):
        """
        Convert [B, w_dim] -> [B, num_ws, w_dim]
        """
        return w.unsqueeze(1).repeat(1, self.num_ws, 1)

    def style_mix(self, ws1, ws2):
        """
        Layer-wise style mixing
        """
        cutoff = random.randint(1, self.num_ws - 1)

        mixed = ws1.clone()
        mixed[:, cutoff:] = ws2[:, cutoff:]

        return mixed

    def forward(self, z, style_mixing_prob=0.3):

        # ==========================================
        # Map Z -> W
        # ==========================================
        w1 = self.mapping(z)                 # [B,w_dim]
        ws1 = self.make_style_bundle(w1)     # [B,L,w_dim]

        # ==========================================
        # Style Mixing Regularization
        # ==========================================
        if self.training and random.random() < style_mixing_prob:

            z2 = torch.randn_like(z)
            w2 = self.mapping(z2)
            ws2 = self.make_style_bundle(w2)

            ws = self.style_mix(ws1, ws2)

        else:
            ws = ws1

        # ==========================================
        # Generate Image
        # ==========================================
        img = self.synthesis(ws)

        return img