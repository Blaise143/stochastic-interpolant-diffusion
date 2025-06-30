import torch.nn as nn
import torch
from typing import Union
import math


class FourrierFeatureEmbedding(nn.Module):
    """
    Based on the encodings from this paper : https://bmild.github.io/fourfeat/
    might ideally work better than sinusoidal embeddings for stochastic interpolants
    """

    def __init__(self, embed_dim: int, scale: float = 30.):
        super().__init__()

        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        self.scale = scale
        self.register_buffer('W', torch.randn(embed_dim//2)*scale)

    def forward(self, t: torch.Tensor):
        if t.ndim == 1:
            t = t[:, None]
        proj = 2*math.pi * t*self.W[None, :]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Sinusoidal_Embedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: Union[torch.Tensor, float]):
        dim = self.embed_dim
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([[t]])
        if len(t.shape) == 1:
            t = t[:, None]
            # t = t.unsqueeze(-1)

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, kind: str = "fourrier"):
        super().__init__()

        assert kind in [
            "fourrier", "sinusoidal"], "Embedding type should be fourrier or sinusoidal"

        if kind == "fourrier":
            self.embedding = FourrierFeatureEmbedding(embed_dim=embed_dim)
        else:
            self.embedding = Sinusoidal_Embedding(embed_dim=embed_dim)

    def forward(self, t: torch.Tensor):
        t = self.embedding(t)
        return t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    gaussian_fourier_embedding = FourrierFeatureEmbedding(64)
    sinusoidal_embedding = Sinusoidal_Embedding(64)
    emb = gaussian_fourier_embedding(torch.tensor(4.))
    emb2 = sinusoidal_embedding(torch.tensor(4.))
    embedding = TimeEmbedding(64)
    emb3 = embedding(torch.tensor(4.))
    print(emb.shape, emb2.shape, emb3.shape)
    print(torch.isclose(emb3, emb))
