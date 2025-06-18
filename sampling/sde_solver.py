import torch
import wandb
# from models import Diffusion


class EulerMaruyamaSolver:
    def __init__(self, model, num_steps: int):
        self.model = model
        self.h = 1/num_steps
        self.num_steps = num_steps
        self.device = next(model.parameters()).device

    def step(self, xt, t):
        eps = torch.randn_like(xt)
        velocity = self.get_drift(xt, t)
        x_next = xt + self.h*velocity + (self.h**0.5)*eps
        return x_next

    def sample_loop(self, shape=(4, 1, 28, 28)):
        xt = torch.randn(shape, device =self.device)
        for i in range(self.num_steps):
            t = torch.full((shape[0],1), 1-i/self.num_steps, device=self.device)
            xt = self.step(xt, t)
        return xt

    def get_drift(self, xt, t):
        eps_pred = self.model(xt, t)
        drift = (xt - eps_pred)/t.view(-1, 1, 1, 1)
        return drift


if __name__ == "__main__":
    model = Diffusion()
    x = torch.rand((3, 1, 28, 28))
    batch_size = x.shape[0]
    t = torch.rand(batch_size, 1)
    eps = model(x, t)
    solver = EulerMaruyamaSolver(model, 10)
    sol = solver.sample_loop((x.shape))
    print(sol.shape)
