import torch
import wandb
from models.flow import Flow


class EulerSolver:
    def __init__(self, model: Flow, num_steps: int):
        self.model = model
        self.h = 1/num_steps
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    def step(self, xt, t):
        """
        One step of Euler's method
        """
        drift = self.model(xt, t)
        x_next = xt + drift*self.h
        return x_next

    def sample_loop(self, shape=(4, 1, 28, 28)):
        xt = torch.randn(shape, device=self.device)
        for i in range(self.num_steps):
            t = torch.full((shape[0], 1), i /
                           self.num_steps, device=self.device)
            xt = self.step(xt, t)
        return xt
