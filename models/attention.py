import torch
import torch.nn as nn


class SelfAttention2d(nn.Module):
    """Spatial self-attention over a 2D feature map.

    Flattens the spatial dimensions into a sequence, applies standard
    multi-head self-attention, and reshapes back. Pairwise attention is
    O((H*W)^2), so this is meant to be used at low resolutions (e.g. a
    U-Net bottleneck) rather than on full-resolution feature maps.

    Attributes:
        channels: Number of input/output channels.
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of channels of the input feature map.
            num_heads: Number of attention heads; must evenly divide `channels`.
        """
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention over spatial positions, with a residual connection.

        Args:
            x: Feature map, shape (batch, channels, H, W).

        Returns:
            Feature map of the same shape, with attention applied.
        """
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(batch, channels, height * width).transpose(1, 2)  # (batch, H*W, channels)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.transpose(1, 2).view(batch, channels, height, width)

        return residual + x
