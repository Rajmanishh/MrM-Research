import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, 1, h, w, device=x.device)
        return x + self.weight * noise