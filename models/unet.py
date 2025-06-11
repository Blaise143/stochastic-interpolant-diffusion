import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from embedding import TimeEmbedding


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 normalization: nn.Module = nn.BatchNorm2d,
                 activation: nn.Module = nn.ReLU(),
                 embedding_dim: int = 100,
                 up_sample=False):
        super().__init__()
        self.up_sample = up_sample
        self.embedding_dim = embedding_dim

        if up_sample:
            self.convs = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2
                ),
                normalization(out_channels),
                activation
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                ),
                normalization(out_channels),
                activation
            )

        if embedding_dim is not None:
            self.time_embedding = TimeEmbedding(
                embed_dim=embedding_dim, projection_featues=out_channels)
        else:
            self.time_embedding = None

    def forward(self, x: torch.Tensor, t_emb) -> torch.Tensor:
        x = self.convs(x)
        if self.time_embedding is not None and t_emb is not None:
            t_proj = self.time_embedding(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + t_proj

        return x


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_list: List[int],
        embedding_dim: int = 100,
        use_sinusoidal: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.initial = nn.ModuleList([
            ConvBlock(
                in_channels,
                channels_list[0],
                stride=1,
                embedding_dim=embedding_dim,
                up_sample=False
            ),
            ConvBlock(
                channels_list[0],
                channels_list[0],
                stride=1,
                embedding_dim=embedding_dim,
                up_sample=False
            )
        ])

        self.downs = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            blocks = nn.ModuleList([
                ConvBlock(
                    channels_list[i],
                    channels_list[i+1],
                    stride=2,
                    embedding_dim=embedding_dim,
                    up_sample=False
                ),
                ConvBlock(
                    channels_list[i+1],
                    channels_list[i+1],
                    stride=1,
                    embedding_dim=embedding_dim,
                    up_sample=False
                )
            ])
            self.downs.append(blocks)

        self.bottleneck = nn.ModuleList([
            ConvBlock(
                channels_list[-1],
                channels_list[-1],
                stride=1,
                embedding_dim=embedding_dim,
                up_sample=False
            ),
            ConvBlock(
                channels_list[-1],
                channels_list[-1],
                stride=1,
                embedding_dim=embedding_dim,
                up_sample=False
            )
        ])

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels_list) - 1)):
            blocks = nn.ModuleList([
                ConvBlock(
                    channels_list[i+1],
                    channels_list[i],
                    up_sample=True,
                    embedding_dim=embedding_dim
                ),
                ConvBlock(
                    channels_list[i] * 2,
                    channels_list[i],
                    stride=1,
                    embedding_dim=embedding_dim,
                    up_sample=False
                ),
                ConvBlock(
                    channels_list[i],
                    channels_list[i],
                    stride=1,
                    embedding_dim=embedding_dim,
                    up_sample=False
                )
            ])
            self.ups.append(blocks)

        self.final = nn.Conv2d(channels_list[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.initial:
            x = block(x, t)
        skips.append(x)
        for i, down_blocks in enumerate(self.downs):
            for block in down_blocks:
                x = block(x, t)
            if i < len(self.downs) - 1:
                skips.append(x)

        for block in self.bottleneck:
            x = block(x, t)

        for up_blocks in self.ups:
            x = up_blocks[0](x, t)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up_blocks[1](x, t)
            x = up_blocks[2](x, t)
        return self.final(x)


if __name__ == "__main__":
    model = Unet(
        in_channels=1,
        out_channels=1,
        channels_list=[32, 64, 128],
        embedding_dim=100,
        use_sinusoidal=False
    )

    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    output = model(x, t)
    print(output.shape, x.shape)
