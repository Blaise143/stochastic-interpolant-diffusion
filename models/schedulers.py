import torch.nn as nn
import torch


class LinearScheduler:
    def __init__(self, t: torch.Tensor):
        self.alpha = t.view(-1, 1, 1, 1)
        self.beta = 1-t.view(-1, 1, 1, 1)

    @property
    def alpha_dot(self):
        return torch.full_like(self.alpha, 1.)

    @property
    def beta_dot(self):
        return torch.full_like(self.beta, -1.)

    def __repr__(self):
        return f"alpha: {self.alpha}, beta: {self.beta}\nalpha_dot: {self.alpha_dot}"


if __name__ == "__main__":
    scheduler = LinearScheduler(4)
    print(scheduler.alpha_dot)
