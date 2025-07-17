import torch
import wandb
from models.flow import Flow
from sampling.base_solver import Solver


class EulerSolver(Solver):

    def step(self, xt, t, labels=None, guidance_scale=None):
        """
        One step of Euler's method
        """
        drift = self.model(xt, t, labels, guidance_scale)
        x_next = xt + drift*self.h
        return x_next
