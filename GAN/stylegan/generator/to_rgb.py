import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToRGB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(3, in_ch, 1, 1)
        )

        self.bias = nn.Parameter(
            torch.zeros(3)
        )

        self.scale = math.sqrt(2 / in_ch)

    def forward(self, x):

        out = F.conv2d(
            x,
            self.weight * self.scale,
            self.bias
        )

        return out