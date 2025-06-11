import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from embedding import TimeEmbedding


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, normalization: nn.BatchNorm2d,
                 activation: nn.ReLU = nn.ReLU()):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class UNET(nn.Module):
    def __init__(self, channels_list: List[int]):
        super().__init__()

        encoder = nn.ModuleList()

    @classmethod
    def sinusoidal_embedding(t: Union[float, torch.Tensor]):
        ...
