import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2 / fan_in)

    def forward(self, x):
        w = self.weight * self.scale
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = math.sqrt(2 / in_features)

    def forward(self, x):
        w = self.weight * self.scale
        return F.linear(x, w, self.bias)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = F.avg_pool2d(x, 2) 
        return x

class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=8):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(b, self.group_size)
        y = x.view(b // group_size, group_size, c, h, w)
        y = torch.var(y, dim=1, unbiased=False)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True) 
        y = y.repeat_interleave(group_size, dim=0) 
        y = y.expand(b, 1, h, w)
        return torch.cat([x, y], dim=1)

class StyleGAN1Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, 32, 1),
            nn.LeakyReLU(0.2)
        )

        # NEW: 4 Blocks down instead of 3
        self.block1 = DiscriminatorBlock(32, 64)   # 64x64 -> 32x32
        self.block2 = DiscriminatorBlock(64, 128)  # 32x32 -> 16x16
        self.block3 = DiscriminatorBlock(128, 256) # 16x16 -> 8x8
        self.block4 = DiscriminatorBlock(256, 256) # 8x8 -> 4x4

        self.mbstd = MinibatchStdDev(group_size=4)
        
        self.final_conv1 = EqualizedConv2d(256 + 1, 256, 3, padding=1)
        self.final_conv2 = EqualizedConv2d(256, 256, 4, padding=0) 
        
        self.flatten = nn.Flatten()
        self.final_linear = EqualizedLinear(256, 1)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.mbstd(x)
        
        x = self.final_conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.final_conv2(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.flatten(x)
        return self.final_linear(x)