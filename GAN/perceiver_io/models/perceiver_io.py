import torch
import torch.nn as nn
from .modules.latent_array import LatentArray
from .modules.cross_attention import CrossAttention
from .modules.self_attention import SelfAttention
from .modules.feed_forward import FeedForward
from .modules.positional_encoding import PositionalEncoding

class PerceiverIO(nn.Module):
    def __init__(self, C, N, D, num_layers=4):
        super().__init__()

        self.latents = LatentArray(N, D)

        self.pos_encoding = PositionalEncoding(max_len=1000, D=C)

        self.cross_attn = CrossAttention(C, D)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(D),
                FeedForward(D)
            ])
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(D)

    def forward(self, x):
        # x: (B, M, C)

        B = x.size(0)

        x = self.pos_encoding(x)

        latents = self.latents(B)

        # Cross-attention (once)
        latents = latents + self.cross_attn(latents, x)

        # Deep latent processing
        for self_attn, ff in self.layers:
            latents = latents + self_attn(latents)
            latents = latents + ff(latents)

        return self.final_norm(latents)