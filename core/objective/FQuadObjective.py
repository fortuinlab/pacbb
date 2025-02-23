import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class FQuadObjective(AbstractObjective):
    """
    A "f-quad" objective from Perez-Ortiz et al. (2021), which involves
    a quadratic expression derived from the PAC-Bayes bound.
    """

    def __init__(self, kl_penalty: float, delta: float):
        """
        Args:
            kl_penalty (float): Coefficient to scale the KL term.
            delta (float): Confidence parameter for the PAC-Bayes bound.
        """
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Compute the f-quad objective.

        This objective calculates:
            ( sqrt(loss + ratio) + sqrt(ratio) )^2
        where ratio = (KL + ln(2 sqrt(n)/delta)) / (2n).

        Args:
            loss (Tensor): Empirical loss.
            kl (Tensor): KL divergence.
            num_samples (float): Dataset size or similar factor.

        Returns:
            Tensor: The scalar objective value.
        """
        kl = kl * self._kl_penalty
        kl_ratio = torch.div(
            kl + np.log((2 * np.sqrt(num_samples)) / self._delta), 2 * num_samples
        )
        first_term = torch.sqrt(loss + kl_ratio)
        second_term = torch.sqrt(kl_ratio)
        return torch.pow(first_term + second_term, 2)
