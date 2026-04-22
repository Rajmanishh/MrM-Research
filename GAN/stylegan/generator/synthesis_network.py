import torch
import torch.nn as nn
import torch.nn.functional as F

from .styled_conv import StyledConv
from .to_rgb import ToRGB


class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim=256):
        super().__init__()

        # number of style inputs
        self.num_ws = 10

        # stronger but stable learned constant
        self.const = nn.Parameter(
            torch.randn(1, 256, 4, 4) * 0.10
        )

        # -------------------------
        # 4x4
        # -------------------------
        self.conv1_4 = StyledConv(256, 256, w_dim)
        self.conv2_4 = StyledConv(256, 256, w_dim)
        self.rgb4 = ToRGB(256)

        # -------------------------
        # 8x8
        # -------------------------
        self.conv1_8 = StyledConv(256, 256, w_dim)
        self.conv2_8 = StyledConv(256, 256, w_dim)
        self.rgb8 = ToRGB(256)

        # -------------------------
        # 16x16
        # -------------------------
        self.conv1_16 = StyledConv(256, 256, w_dim)
        self.conv2_16 = StyledConv(256, 256, w_dim)
        self.rgb16 = ToRGB(256)

        # -------------------------
        # 32x32
        # -------------------------
        self.conv1_32 = StyledConv(256, 128, w_dim)
        self.conv2_32 = StyledConv(128, 128, w_dim)
        self.rgb32 = ToRGB(128)

        # -------------------------
        # 64x64
        # -------------------------
        self.conv1_64 = StyledConv(128, 64, w_dim)
        self.conv2_64 = StyledConv(64, 64, w_dim)
        self.rgb64 = ToRGB(64)

    def upsample(self, x):
        return F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

    def forward(self, ws):

        b = ws.shape[0]

        # learned constant input
        x = self.const.repeat(b, 1, 1, 1)

        # =====================================
        # 4x4
        # =====================================
        x = self.conv1_4(x, ws[:, 0])
        x = self.conv2_4(x, ws[:, 1])
        rgb = self.rgb4(x)

        # =====================================
        # 8x8
        # =====================================
        x = self.upsample(x)
        rgb = self.upsample(rgb)

        x = self.conv1_8(x, ws[:, 2])
        x = self.conv2_8(x, ws[:, 3])

        rgb = rgb + 0.5 * self.rgb8(x)

        # =====================================
        # 16x16
        # =====================================
        x = self.upsample(x)
        rgb = self.upsample(rgb)

        x = self.conv1_16(x, ws[:, 4])
        x = self.conv2_16(x, ws[:, 5])

        rgb = rgb + 0.5 * self.rgb16(x)

        # =====================================
        # 32x32
        # =====================================
        x = self.upsample(x)
        rgb = self.upsample(rgb)

        x = self.conv1_32(x, ws[:, 6])
        x = self.conv2_32(x, ws[:, 7])

        rgb = rgb + 0.5 * self.rgb32(x)

        # =====================================
        # 64x64
        # =====================================
        x = self.upsample(x)
        rgb = self.upsample(rgb)

        x = self.conv1_64(x, ws[:, 8])
        x = self.conv2_64(x, ws[:, 9])

        rgb = rgb + 0.5 * self.rgb64(x)

        return rgb