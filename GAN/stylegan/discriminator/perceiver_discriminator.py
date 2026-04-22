import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from perceiver_io.models.perceiver_io import PerceiverIO


class PerceiverDiscriminator(nn.Module):
    """
    Updated Hybrid CNN + Perceiver Discriminator
    Optimized for GAN stability + lower FID

    Input : [B,3,64,64]
    Output: [B,1]
    """

    def __init__(
        self,
        img_size=64,
        patch_size=2,
        token_dim=128,
        latent_dim=256,
        num_latents=128,
        num_layers=8
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # ==================================================
        # CNN STEM (Strong Local Feature Extractor)
        # ==================================================
        self.stem = nn.Sequential(

            spectral_norm(
                nn.Conv2d(3, 64, 3, 1, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(64, 128, 4, 2, 1)
            ),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(128, 128, 3, 1, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(128, 128, 3, 1, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ==================================================
        # After stem: 32x32 feature map
        # Patchify with patch_size=2 => 16x16 = 256 tokens
        # ==================================================
        feat_size = img_size // 2
        self.num_patches = (feat_size // patch_size) ** 2

        patch_dim = 128 * patch_size * patch_size

        self.input_proj = nn.Linear(
            patch_dim,
            token_dim
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, token_dim) * 0.02
        )

        # ==================================================
        # Perceiver Backbone
        # ==================================================
        self.backbone = PerceiverIO(
            C=token_dim,
            N=num_latents,
            D=latent_dim,
            num_layers=num_layers
        )

        # ==================================================
        # Attention Pooling
        # ==================================================
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1)
        )

        # ==================================================
        # Final GAN Head
        # ==================================================
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),

            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),

            nn.Linear(latent_dim // 2, 1)
        )

    # ======================================================
    # Patchify helper
    # ======================================================
    def patchify(self, x):
        """
        x: [B,C,H,W]
        -> [B,num_tokens,C*p*p]
        """
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.contiguous().view(B, -1, C * p * p)

        return x

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x):

        # -----------------------------------------
        # CNN local features
        # -----------------------------------------
        x = self.stem(x)                 # [B,128,32,32]

        # -----------------------------------------
        # Patch tokens
        # -----------------------------------------
        x = self.patchify(x)            # [B,T,patch_dim]

        # -----------------------------------------
        # Token projection + positions
        # -----------------------------------------
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1)]

        # -----------------------------------------
        # Perceiver global reasoning
        # -----------------------------------------
        latents = self.backbone(x)      # [B,N,D]

        # -----------------------------------------
        # Learned attention pooling
        # -----------------------------------------
        attn = torch.softmax(
            self.attn_pool(latents),
            dim=1
        )

        pooled = (latents * attn).sum(dim=1)

        # -----------------------------------------
        # Real/Fake score
        # -----------------------------------------
        out = self.head(pooled)

        return out