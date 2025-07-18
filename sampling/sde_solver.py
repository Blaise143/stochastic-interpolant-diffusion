import torch
from sampling.base_solver import Solver

class EulerMaruyamaSolver(Solver):
    def step(self, xt, t, labels=None, guidance_scale=None):
        """
        One step of Euler Maruyama method
        """
        eps = torch.randn_like(xt)
        drift = self.model(xt, t, labels, guidance_scale)
        x_next = xt + drift*self.h + (self.h**0.5)*eps
        return x_next
