import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, D, dropout=0.1):
        super().__init__()

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)

        self.norm = nn.LayerNorm(D)

        self.out_proj = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

        self.D = D

    def forward(self, x):
        # x: (B, N, D)

        x_norm = self.norm(x)

        Q = self.Wq(x_norm)
        K = self.Wk(x_norm)
        V = self.Wv(x_norm)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)  # (B, N, N)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = self.out_proj(out)

        return out