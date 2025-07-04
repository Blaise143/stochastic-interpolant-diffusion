import torch
import torch.nn as nn
from models.embedding import TimeEmbedding
from typing import List
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        normalization: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU(),
        embedding_dim: int = 100,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

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


class NoisyClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels_list: List[int] = [32, 64, 128],
        embedding_dim: int = 100,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_embedding = TimeEmbedding(embed_dim=embedding_dim)
        self.num_classes = num_classes

        self.initial = ConvBlock(
            in_channels,
            channels_list[0],
            stride=1,
            embedding_dim=embedding_dim,
        )

        self.downs = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.downs.append(
                ConvBlock(
                    channels_list[i],
                    channels_list[i+1],
                    stride=2,
                    embedding_dim=embedding_dim,
                )
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels_list[-1], num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x = self.initial(x, t_emb)
        for down in self.downs:
            x = down(x, t_emb)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self(x, t)
        return F.cross_entropy(logits, labels)

    def get_guidance(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gradient of log p(y|x,t) wrt x
        """
        # x.requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)

        logits = self(x, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, y.unsqueeze(1))

        grad_outputs = torch.ones_like(selected_log_probs)
        grad_x, = torch.autograd.grad(
            outputs=selected_log_probs,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )

        x.requires_grad_(False)
        return grad_x


if __name__ == "__main__":
    model = NoisyClassifier()
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    t = torch.rand(2, 1)
    y = torch.tensor([1, 3])

    grad = model.get_guidance(x, t, y)
    print("Gradient shape:", grad.shape)
    print("Gradient stats: min =", grad.min().item(), "max =",
          grad.max().item(), "mean =", grad.mean().item())
