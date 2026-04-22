import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, D):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(max_len, D) * 0.02)

    def forward(self, x):
        # x: (B, M, D)
        B, M, _ = x.shape

        pos = self.pos_embed[:M].to(x.device)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        return x + pos