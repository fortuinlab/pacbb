import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class TolstikhinObjective(AbstractObjective):
    """
    Objective related to Tolstikhin et al. (2013), featuring a combination of
    the empirical loss, a square-root term involving KL, and an additional additive term.
    """
    def __init__(self, kl_penalty: float, delta: float):
        """
        Args:
            kl_penalty (float): The coefficient multiplying the KL term.
            delta (float): Confidence parameter in the PAC-Bayes or related bound.
        """
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Compute the Tolstikhin objective.

        The final expression includes:
            loss + sqrt(2 * loss * ratio) + 2 * ratio
        where ratio = (kl_penalty*KL + ln(2n) - ln(delta)) / (2n).

        Args:
            loss (Tensor): Empirical loss.
            kl (Tensor): KL divergence.
            num_samples (float): Number of data samples for normalization.

        Returns:
            Tensor: The scalar objective value.
        """
        kl = kl * self._kl_penalty
        second_term = 2 * loss * torch.div(kl + np.log(2 * num_samples) - np.log(self._delta),
                                           2 * num_samples)
        third_term = 2 * torch.div(kl + np.log(2 * num_samples) - np.log(self._delta),
                                   2 * num_samples)
        return loss + torch.sqrt(second_term) + third_term
