import torch
import torch.nn as nn
from models.stochastic_interpolant import StochasticInterpolant
from models.unet import Unet


class Flow(StochasticInterpolant):
    def __init__(self,  model=Unet(in_channels=1,
                                   out_channels=1,
                                   channels_list=[32, 64, 128],
                                   embedding_dim=100,)):
        super().__init__()
        self.vector_field = model
        self.criterion = nn.MSELoss()

    def sample(self, x: torch.Tensor):
        """
        Returns xt from the probability path

        Args:
            x (torch.Tensor): Input sample
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
        return xt, t, target_vector_field

    def get_vector_field(self, x):
        return self.vector_field(xt, t)

    def forward(self, x, t=None):
        if t is None:
            xt, t, eps = self.sample(x)
            return self.vector_field(xt, t)
        else:
            return self.vector_field(x, t)

    def compute_loss(self, x):

        xt, t, target_vector_field = self.sample(x)
        pred = self.vector_field(xt, t)
        loss = self.criterion(pred, target_vector_field)
        return loss
