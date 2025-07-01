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
            guidance_scale = self.guidance_scale

        if labels is not None and guidance_scale > 0:
            unguided = self.vector_field(x, t)
            guided = self.vector_field(x, t, labels)
            if random.random() < 0.1:
                guidance_scale = 0
            return unguided + guidance_scale*(guided-unguided)
        else:
            return self.vector_field(x, t, labels)

    def forward(self, x, t=None, labels=None, guidance_scale=None):
        if t is None:
            xt, t, target_vector_field, labels = self.sample(x, labels)
            # return self.vector_field(xt, t)
            return self.get_vector_field(xt, t, labels, guidance_scale)
        else:
            # return self.vector_field(x, t)
            return self.get_vector_field(x, t, labels, guidance_scale)

    def compute_loss(self, x, labels=None):

        xt, t, target_vector_field, labels = self.sample(x, labels)
        pred = self.vector_field(xt, t, labels)
        loss = self.criterion(pred, target_vector_field)
        return loss
