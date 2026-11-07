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
        """Draws a training pair from the probability path at a random time.

        Args:
            x: Input data tensor.
            labels: Optional target labels, for conditional interpolants.

        Returns:
            A tuple (xt, t, target_vector_field, labels): the interpolated
            point, the sampled time, the regression target for the vector
            field, and the labels (possibly None).

        Raises:
            NotImplementedError: If the subclass does not implement this.
        """
        raise NotImplementedError("Subclasses must implement sample()")

    @abstractmethod
    def compute_loss(self, x: torch.Tensor):
        """Computes the training loss for a batch of data.

        Args:
            x: Input data tensor.

        Raises:
            NotImplementedError: If the subclass does not implement this.
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")
