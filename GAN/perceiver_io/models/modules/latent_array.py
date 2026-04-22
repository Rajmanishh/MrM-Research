import torch
import torch.nn as nn
class LatentArray(nn.Module):
    def __init__(self, N, D):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(N, D)*0.02)

    def forward(self, batch_size):
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)