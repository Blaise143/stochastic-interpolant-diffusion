import torch
from torch import Tensor
import math


class LinearScheduler:
    """Implements the linear condOT path

    Attributes:
        alpha: t
        beta: 1-t
    """

    def __init__(self, t: Tensor) -> None:
        self.alpha = t.view(-1, 1, 1, 1)
        self.beta = 1 - t.view(-1, 1, 1, 1)

    @property
    def alpha_dot(self) -> Tensor:
        """Derivative of alpha with respect to t

        Returns:
            1
        """
        return torch.full_like(self.alpha, 1.0)

    @property
    def beta_dot(self) -> Tensor:
        """Derivative of beta_t with respect to t

        Returns:
            -1
        """
        return torch.full_like(self.beta, -1.0)

    def __repr__(self) -> str:
        return f"alpha: {self.alpha}, beta: {self.beta}\nalpha_dot: {self.alpha_dot}"


class TrigonometricScheduler:
    """Implements a trigonometric (sine/cosine) probability path / interpolant.

    Attributes:
        alpha: sin(pi/2 * t)
        beta: cos(pi/2 * t)
    """

    def __init__(self, t: Tensor) -> None:
        t_view = t.view(-1, 1, 1, 1)
        self.alpha = torch.sin(math.pi / 2 * t_view)
        self.beta = torch.cos(math.pi / 2 * t_view)
        self._t = t_view

    @property
    def alpha_dot(self) -> Tensor:
        """Derivative of alpha with respect to t: (pi/2) * cos(pi/2 * t)"""
        return (math.pi / 2) * torch.cos(math.pi / 2 * self._t)

    @property
    def beta_dot(self) -> Tensor:
        """Derivative of beta with respect to t: -(pi/2) * sin(pi/2 * t)"""
        return -(math.pi / 2) * torch.sin(math.pi / 2 * self._t)

    def __repr__(self) -> str:
        return f"TrigonometricScheduler(alpha=sin(pi/2*t), beta=cos(pi/2*t))"


class PolynomialScheduler:
    """Implements a polynomial-curvature probability path / interpolant.

    p=1 reduces to the linear condOT path; p>1 spends more of the
    path near the endpoints and less in the middle.

    Attributes:
        alpha: t**p
        beta: (1-t)**p
    """

    p: float = 2.0

    def __init__(self, t: Tensor) -> None:
        t_view = t.view(-1, 1, 1, 1)
        self.alpha = t_view**self.p
        self.beta = (1 - t_view) ** self.p
        self._t = t_view

    @property
    def alpha_dot(self) -> Tensor:
        """Derivative of alpha with respect to t: p * t**(p-1)"""
        return self.p * self._t ** (self.p - 1)

    @property
    def beta_dot(self) -> Tensor:
        """Derivative of beta with respect to t: -p * (1-t)**(p-1)"""
        return -self.p * (1 - self._t) ** (self.p - 1)

    def __repr__(self) -> str:
        return f"PolynomialScheduler(p={self.p}, alpha=t**{self.p}, beta=(1-t)**{self.p})"


if __name__ == "__main__":
    scheduler = LinearScheduler(torch.tensor(0.5))
    print(scheduler.alpha_dot)
    trig_scheduler = TrigonometricScheduler(torch.tensor(0.5))
    print(trig_scheduler.alpha, trig_scheduler.alpha_dot)
    poly_scheduler = PolynomialScheduler(torch.tensor(0.5))
    print(poly_scheduler.alpha, poly_scheduler.alpha_dot)
