import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod

class StochasticInterpolant(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def sample(self, x: torch.Tensor):
        ...

    @abstractmethod
    def compute_loss(self, x: torch.Tensor):
        ...

