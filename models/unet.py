import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from models.embedding import TimeEmbedding, LabelEmbedding


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
            self.time_projection = nn.Linear(embedding_dim, out_channels)

        else:
            self.time_projection = None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        if self.time_projection is not None and t_emb is not None:
            t_proj = self.time_projection(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + t_proj

        return x


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_list: List[int],
        embedding_dim: int = 100,
        num_classes: int = 10
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_embedding = TimeEmbedding(embed_dim=embedding_dim)
        self.label_embedding = LabelEmbedding(10, embedding_dim)

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
                up_sample=False,
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:

        t_emb = self.time_embedding(t)

        if labels is not None:
            label_emb = self.label_embedding(labels)
            combined_embeddings = t_emb + label_emb
        else:
            combined_embeddings = t_emb
        skips = []
        for block in self.initial:
            x = block(x, combined_embeddings)

        skips.append(x)
        for i, down_blocks in enumerate(self.downs):
            for block in down_blocks:
                x = block(x, combined_embeddings)
            if i < len(self.downs) - 1:
                skips.append(x)

        for block in self.bottleneck:
            x = block(x, combined_embeddings)

        for up_blocks in self.ups:
            x = up_blocks[0](x, combined_embeddings)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up_blocks[1](x, combined_embeddings)
            x = up_blocks[2](x, combined_embeddings)
        return self.final(x)


if __name__ == "__main__":
    model = Unet(
        in_channels=1,
        out_channels=1,
        channels_list=[32, 64, 128],
        embedding_dim=100,
    )

    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    output = model(x, t)
    print(output.shape, x.shape)

    output1 = model(x, t)

    labels = torch.randint(0, 10, (4,))
    output2 = model(x, t, labels)

    print("Unconditional output shape:", output1.shape)
    print("Conditional output shape:", output2.shape)
    print("Input shape:", x.shape)
