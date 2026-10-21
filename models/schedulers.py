import torch
from torch import Tensor


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


if __name__ == "__main__":
    scheduler = LinearScheduler(torch.tensor(23))
    print(scheduler.alpha_dot)
