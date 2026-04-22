import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, C, D, dropout=0.1):
        super().__init__()

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(C, D)
        self.Wv = nn.Linear(C, D)

        self.norm_q = nn.LayerNorm(D)
        self.norm_kv = nn.LayerNorm(C)

        self.out_proj = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

        self.D = D

    def forward(self, latents, inputs):
        # latents: (B, N, D)
        # inputs:  (B, M, C)

        Q = self.Wq(self.norm_q(latents))   # (B, N, D)
        K = self.Wk(self.norm_kv(inputs))   # (B, M, D)
        V = self.Wv(self.norm_kv(inputs))   # (B, M, D)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)  # (B, N, M)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, N, D)
        out = self.out_proj(out)

        return out