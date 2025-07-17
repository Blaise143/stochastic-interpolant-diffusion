from abc import abstractmethod, ABC
from models.flow import Flow
import torch

class Solver(ABC):
    def __init__(self, model: Flow, num_steps: int):
        self.model = model
        self.h = 1/num_steps
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    @abstractmethod
    def step(self, xt, t, labels=None, guidance_scale=None):
        return ...

    def sample_loop(self, shape=(4, 1, 28, 28), labels=None, guidance_scale=None):
        xt = torch.randn(shape, device=self.device)
        for i in range(self.num_steps):
            t = torch.full((shape[0], 1), i /
                           self.num_steps, device=self.device)
            xt = self.step(xt, t, labels, guidance_scale)
        return xt
