import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, D, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(D)

        # 🔥 Two projections for SwiGLU
        self.w1 = nn.Linear(D, 4 * D)
        self.w2 = nn.Linear(D, 4 * D)

        self.proj = nn.Linear(4 * D, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        # 🔥 SwiGLU
        x = self.w1(x) * F.silu(self.w2(x))

        x = self.dropout(x)
        x = self.proj(x)
        x = self.dropout(x)

        return x