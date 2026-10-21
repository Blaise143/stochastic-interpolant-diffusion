import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class StochasticInterpolant(nn.Module, ABC):
    """
    Base class for stochastic interpolant for both a flow model and a diffusion model
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def sample(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> tuple:
        """Abstract method for sampling a probability path

        Args:
            x:Input tensor
            label: the target label, by default none

        Returns:


        Raises:
            NotImplementedError:
        """
        raise NotImplementedError("Implement This bri")

    @abstractmethod
    def compute_loss(self, x: torch.Tensor):
        raise NotImplementedError("Implement this bro ")
