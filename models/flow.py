from typing import Optional

import torch
import torch.nn as nn
from models.stochastic_interpolant_base import StochasticInterpolant
from models.unet import Unet
from models.schedulers import LinearScheduler, TrigonometricScheduler, PolynomialScheduler

SCHEDULERS = {
    "linear": LinearScheduler,
    "trigonometric": TrigonometricScheduler,
    "polynomial": PolynomialScheduler,
}


class Flow(StochasticInterpolant):
    """A flow matching class.

    Attributes:
        vector_field: The model parametrizing the vector field
        criterion: The loss function for supervised learning
        guidance_scale: the guidance scale for classifier free guidance
        scheduler_type: the probability path scheduler ('linear' or 'trigonometric')
    """

    def __init__(
        self,
        model=None,
        num_classes=10,
        guidance_scale=7.0,
        scheduler_type="linear",
        use_attention=False,
    ):
        """
        Args:
            model: The vector field network; if None, builds the default
                `Unet` (using `num_classes` and `use_attention` below).
            num_classes: Number of classes, used only when `model` is None.
            guidance_scale: Default classifier-free guidance scale.
            scheduler_type: Probability path scheduler ('linear',
                'trigonometric', or 'polynomial').
            use_attention: If True, the default `Unet` gets a bottleneck
                self-attention block. Ignored if `model` is given explicitly.
        """
        super().__init__()
        if scheduler_type not in SCHEDULERS:
            raise ValueError(
                f"Unknown scheduler_type {scheduler_type!r}, expected one of {list(SCHEDULERS)}"
            )
        if model is None:
            model = Unet(
                in_channels=1,
                out_channels=1,
                channels_list=[32, 64, 128],
                embedding_dim=100,
                num_classes=num_classes,
                use_attention=use_attention,
            )
        self.vector_field = model
        self.criterion = nn.MSELoss()
        self.guidance_scale = guidance_scale
        self.scheduler_type = scheduler_type

    def sample(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
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
        scheduler = SCHEDULERS[self.scheduler_type](t)
        xt = scheduler.alpha * x + scheduler.beta * eps

        target_vector_field = scheduler.alpha_dot * x + scheduler.beta_dot * eps
        return xt, t, target_vector_field, labels

    def get_vector_field(self, x, t, labels=None, guidance_scale=None):
        if labels is None:
            return self.vector_field(x, t, labels)

        if not guidance_scale:
            return self.vector_field(x, t)

        unguided = self.vector_field(x, t)
        guided = self.vector_field(x, t, labels)
        return (1 - guidance_scale) * unguided + guidance_scale * guided

    def forward(self, x, t=None, labels=None, guidance_scale=None):
        """Predicts the vector field, either at a fresh training point or at (x, t).

        If `t` is None, draws a training interpolant from `x` via `sample`
        and evaluates the vector field there; otherwise treats `x` as
        already being `xt` at the given `t` (the path used by the ODE/SDE
        solvers during generation).

        Args:
            x: Data tensor (if `t` is None) or the current state `xt`.
            t: Time, shape (batch, 1), or None to sample one.
            labels: Optional class labels for conditioning.
            guidance_scale: Optional classifier-free guidance scale.

        Returns:
            The predicted vector field at the resulting (x, t).
        """
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

    def get_score(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        labels=None,
        guidance_scale=None,
        sigma=0.6,
    ):
        if t is None:
            _, t, _, labels = self.sample(x, labels)

        u_t = self(x, t, labels, guidance_scale)

        scheduler = SCHEDULERS[self.scheduler_type](t)
        alpha_t = scheduler.alpha
        beta_t = scheduler.beta
        dot_alpha_t = scheduler.alpha_dot
        dot_beta_t = scheduler.beta_dot

        sigma_t = torch.full_like(alpha_t, sigma)
        C_t = (
            (beta_t**2 / alpha_t) * dot_alpha_t - dot_beta_t * beta_t + (sigma_t**2) / 2
        )

        score = (1 / C_t) * (u_t - (dot_alpha_t / alpha_t) * x)

        return score
