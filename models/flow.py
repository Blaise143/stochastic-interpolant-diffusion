import torch
import torch.nn as nn
from models.stochastic_interpolant import StochasticInterpolant
from models.unet import Unet
import random


class Flow(StochasticInterpolant):
    def __init__(self,  model=Unet(in_channels=1,
                                   out_channels=1,
                                   channels_list=[32, 64, 128],
                                   embedding_dim=100,),
                 num_classes=10,
                 guidance_scale=7.):
        super().__init__()
        self.vector_field = model
        self.criterion = nn.MSELoss()
        self.guidance_scale = guidance_scale

    def sample(self, x: torch.Tensor, labels=None):
        """
        Returns xt from the probability path

        Args:
            x (torch.Tensor): Input sample
            labels: class labels
        """
        device = x.device
        batch_size = x.shape[0]
        t = torch.rand(batch_size, 1, device=device)
        eps = torch.randn_like(x)
        t_broadcast = t.view(-1, 1, 1, 1)
        alpha = t_broadcast
        beta = 1-t_broadcast
        xt = alpha*x + beta*eps

        target_vector_field = x - eps
        return xt, t, target_vector_field, labels

    def get_vector_field(self, x, t, labels=None, guidance_scale=None):
        if guidance_scale is None:
            guidance_scale = 0

        if labels is not None:
            unguided = self.vector_field(x, t)
            guided = self.vector_field(x, t, labels)
            return (1-guidance_scale)*unguided + guidance_scale*guided
        else:
            return self.vector_field(x, t, labels)

    def forward(self, x, t=None, labels=None, guidance_scale=None):
        if t is None:
            xt, t, target_vector_field, labels = self.sample(x, labels)
            return self.get_vector_field(xt, t, labels, guidance_scale)
        else:
            return self.get_vector_field(x, t, labels, guidance_scale)

    def compute_loss(self, x, labels=None):

        xt, t, target_vector_field, labels = self.sample(x, labels)
        pred = self.vector_field(xt, t, labels)
        loss = self.criterion(pred, target_vector_field)
        return loss

    def get_score(self, x: torch.Tensor, t: torch.Tensor=None, labels=None, guidance_scale=None, sigma = 0.6):
        u_t = self(x, t, labels, guidance_scale)

        alpha_t = t.view(-1, 1, 1, 1)
        beta_t = 1 - alpha_t

        dot_alpha_t = torch.ones_like(alpha_t)
        dot_beta_t = -torch.ones_like(beta_t)

        sigma_t = torch.full_like(alpha_t, sigma)
        C_t = (beta_t**2 / alpha_t) * dot_alpha_t - dot_beta_t * beta_t + (sigma_t**2) / 2

        score = (1 / C_t) * (u_t - (dot_alpha_t / alpha_t) * x)

        return score
