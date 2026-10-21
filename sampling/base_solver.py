from abc import abstractmethod, ABC
from models.flow import Flow
import torch
from torch import Tensor


class Solver(ABC):
    """Base class for ODE or SDE solver.

    Attributes:
        model: vector field or score matching model
        h: step size for Euler or Euler Maruyama solver
        num_steps: Number of time steps for the solver
        device: one of Cuda, CPU or MPS
    """

    def __init__(self, model: Flow, num_steps: int):
        self.model = model
        self.h = 1 / num_steps
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    @abstractmethod
    def step(self, xt: Tensor, t: Tensor, labels=None, guidance_scale=None):
        return ...

    def sample_loop(self, shape=(4, 1, 28, 28), labels=None, guidance_scale=None):
        xt = torch.randn(shape, device=self.device)
        for i in range(self.num_steps):
            t = torch.full((shape[0], 1), i / self.num_steps, device=self.device)
            xt = self.step(xt, t, labels, guidance_scale)
        return xt
