import torch.nn as nn
import torch
import math


class FourrierFeatureEmbedding(nn.Module):
    """
    Based on the encodings from this paper : https://bmild.github.io/fourfeat/
    might ideally work better than sinusoidal embeddings for stochastic interpolants
    """

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()

        assert embed_dim % 2 == 0

        self.embed_dim = embed_dim
        self.scale = scale
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor):
        """Computes the random Fourier feature embedding of `t`.

        Args:
            t: Time value(s), shape (batch,) or (batch, 1).

        Returns:
            Embedding of shape (batch, embed_dim).
        """
        if t.ndim == 1:
            t = t[:, None]
        proj = 2 * math.pi * t * self.W[None, :]  # pyright: ignore
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Sinusoidal_Embedding(nn.Module):
    """Transformer-style sinusoidal embeddings, used for time.

    Attributes:
        embed_dim: The length of the embedding vectors.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: The embedding dimension to produce.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor | float) -> torch.Tensor:
        """Computes the sinusoidal embedding of `t`.

        Args:
            t: Time value(s): a scalar, or a tensor of shape (batch,) or
                (batch, 1).

        Returns:
            Embedding of shape (batch, embed_dim).
        """
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
        """
        Time embedding module. Depending on `kind`, a fourrier or sinusoidal embeding is used
        Args:
            embed_dim: Embedding dimension
            kind: kind of embedding to be used
        """
        super().__init__()

        assert kind in ["fourrier", "sinusoidal"], (
            "Embedding type should be fourrier or sinusoidal"
        )

        if kind == "fourrier":
            self.embedding = FourrierFeatureEmbedding(embed_dim=embed_dim)
        else:
            self.embedding = Sinusoidal_Embedding(embed_dim=embed_dim)

    def forward(self, t: torch.Tensor):
        """Computes the time embedding of `t` using the configured `kind`.

        Args:
            t: Time value(s), shape (batch,) or (batch, 1).

        Returns:
            Embedding of shape (batch, embed_dim).
        """
        t = self.embedding(t)
        return t


class LabelEmbedding(nn.Module):
    """Embedding for the labels to be used in classifier free guidance

    Attributes:
        embedding: label embeddings
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embed_dim
        )

    def forward(self, labels: torch.Tensor):
        """
        labels shape: [batch_size]
        """
        return self.embedding(labels)


if __name__ == "__main__":
    gaussian_fourier_embedding = FourrierFeatureEmbedding(64)
    sinusoidal_embedding = Sinusoidal_Embedding(64)
    emb = gaussian_fourier_embedding(torch.tensor(4.0))
    emb2 = sinusoidal_embedding(torch.tensor(4.0))
    embedding = TimeEmbedding(64)
    emb3 = embedding(torch.tensor(4.0))
    print(emb.shape, emb2.shape, emb3.shape)
    print(torch.isclose(emb3, emb))
    label_embedding = LabelEmbedding(num_classes=10, embed_dim=64)
    labels = torch.tensor([0, 1, 2, 3])
    emb4 = label_embedding(labels)
    print(emb4.shape)
