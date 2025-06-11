import torch.nn as nn
import torch
from typing import Union
import math


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, projection_featues: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linears = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projection_featues),
            nn.ReLU()
        )

    def forward(self, t):
        t = self.sinusoidal_embedding(t)
        return self.linears(t)

    def sinusoidal_embedding(self, t: Union[float, torch.Tensor]):
        dim = self.embed_dim
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([[t]])
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


if __name__ == "__main__":
    embedding = TimeEmbedding(30, 100)
    out = embedding(0)
    print(out.shape)
