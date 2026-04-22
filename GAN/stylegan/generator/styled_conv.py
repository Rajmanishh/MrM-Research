import torch
import torch.nn as nn
from .noise_injection import NoiseInjection
import numpy as np

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data)
        self.scale = np.sqrt(2 / in_dim)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale, self.bias)

class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        # Equalized Learning Rate for Conv
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv.weight.data.normal_()
        self.conv.bias.data.zero_()
        self.scale = np.sqrt(2 / (in_ch * 3 * 3))
        
        # Style transformation (A) - Learned affine transform
        self.style = EqualizedLinear(w_dim, out_ch * 2) 

        self.noise = NoiseInjection(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        # 1. Convolution with runtime scaling (Equalized LR)
        x = nn.functional.conv2d(x, self.conv.weight * self.scale, self.conv.bias, padding=1)
        
        # 2. Noise Injection (B)
        x = self.noise(x)
        x = self.act(x)

        # 3. AdaIN
        # Calculate instance stats
        mean = x.mean([2, 3], keepdim=True)
        std = x.std([2, 3], keepdim=True) + 1e-8
        x = (x - mean) / std

        # Generate style parameters
        style = self.style(w).unsqueeze(2).unsqueeze(3)
        style_scale, style_bias = style.chunk(2, 1)

        return x * style_scale + style_bias