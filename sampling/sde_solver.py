import torch
# from models.flow import Flow
from sampling.base_solver import Solver

class EulerMaruyamaSolver(Solver):
    # def __init__(self, model: Flow, num_steps: int):
    #     self.model = model
    #     self.h = 1/num_steps
    #     self.num_steps = num_steps
    #     self.device = next(model.parameters()).device

    def step(self, xt, t, labels=None, guidance_scale=None):
        """
        One step of Euler Maruyama method
        """
        eps = torch.randn_like(xt)
        drift = self.model(xt, t, labels, guidance_scale)
        x_next = xt + drift*self.h + (self.h**0.5)*eps
        return x_next

    # def sample_loop(self, shape=(4, 1, 28, 28), labels=None, guidance_scale=None):
    #     xt = torch.randn(shape, device=self.device)
    #     for i in range(self.num_steps):
    #         t = torch.full((shape[0], 1), i /
    #                        self.num_steps, device=self.device)
    #         xt = self.step(xt, t, labels, guidance_scale)
    #     return xt
