import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class FClassicObjective(AbstractObjective):
    """
    The "f-classic" objective from Perez-Ortiz et al. (2021), which
    combines empirical loss with a square-root bounding term involving KL and delta.

    Typically used to ensure a PAC-Bayes bound is minimized during training.
    """
    def __init__(self, kl_penalty: float, delta: float):
        """
        Args:
            kl_penalty (float): Coefficient for scaling KL divergence.
            delta (float): Confidence parameter for the PAC-Bayes bound.
        """
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Compute the f-classic objective.

        Args:
            loss (Tensor): Empirical risk (e.g., average loss on a mini-batch).
            kl (Tensor): KL divergence between posterior and prior.
            num_samples (float): Number of samples or an equivalent scaling factor.

        Returns:
            Tensor: A scalar objective = loss + sqrt( (KL * kl_penalty + ln(2 sqrt(n)/delta)) / (2n) ).
        """
        kl = kl * self._kl_penalty
        kl_ratio = torch.div(kl + np.log((2 * np.sqrt(num_samples)) / self._delta), 2 * num_samples)
        return loss + torch.sqrt(kl_ratio)
