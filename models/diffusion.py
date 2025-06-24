import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.stochastic_interpolant import StochasticInterpolant
from models.unet import Unet


class Diffusion(StochasticInterpolant):
    def __init__(self, model: Unet = Unet(in_channels=1,
                                          out_channels=1,
                                          channels_list=[32, 64, 128],
                                          embedding_dim=100,
                                          use_sinusoidal=False)):
        super().__init__()

        self.eps_predictor = model
        self.criterion = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def sample(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device  # next(self.model.parameters()).device#x.device
        t = torch.rand(batch_size, 1, device=device)
        eps = torch.randn_like(x)

        alpha_t = t.view(batch_size, 1, 1, 1)
        beta_t = 1-alpha_t
        xt = alpha_t*x + beta_t*eps
        return xt, t, eps

    def compute_loss(self, x: torch.Tensor):
        xt, t, eps = self.sample(x)
        eps_pred = self.eps_predictor(xt, t)
        # loss = self.criterion(eps, eps_pred)
        loss = self.criterion(eps_pred, eps)
        return loss

    def forward(self, xt: torch.Tensor, t: torch.Tensor):
        noise_pred = self.eps_predictor(xt, t)
        return noise_pred


if __name__ == '__main__':
    # model = Unet#(in_channels=1, out_channels=1, channels_list=[32, 64, 128],embedding_dim=100, use_sinusoidal=False)
    diff = Diffusion()
    x = torch.randn(32, 1, 28, 28)

    out, t, eps = diff.sample(x)
    print(t.shape)
    # exit()
    print(out.shape, t.shape)
    y = diff(out, t)
    print(y.shape, t.shape, eps.shape, x.shape)
