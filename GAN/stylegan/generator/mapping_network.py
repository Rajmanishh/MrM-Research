import torch
import torch.nn as nn
from .styled_conv import EqualizedLinear

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, depth=8):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(EqualizedLinear(z_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # Pixel-wise normalization
        z = z * torch.rsqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        return self.net(z)