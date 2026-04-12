import torch
import torch.nn as nn
import math


class ClassificationHead(nn.Module):
    def __init__(self, D, num_classes, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.scale = math.sqrt(D)   # 🔥 explicit √D

        self.attn = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, num_heads)
        )

        self.dropout = nn.Dropout(dropout)

        # 🔥 Learnable head combination
        self.head_proj = nn.Linear(num_heads * D, D)

        self.proj = nn.Linear(D, num_classes)

    def forward(self, latents):
        # latents: (B, N, D)

        attn_scores = self.attn(latents)   # (B, N, H)

        # 🔥 Proper √D scaling
        attn_scores = attn_scores / self.scale

        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        x = torch.einsum("bnd,bnh->bhd", latents, attn_weights)

        # 🔥 Learnable combination
        B, H, D = x.shape
        x = x.reshape(B, H * D)
        x = self.head_proj(x)

        logits = self.proj(x)

        return logits