from torch import Tensor

from core.objective import AbstractObjective


class BBBObjective(AbstractObjective):
    """
    The Bayes By Backprop (BBB) objective from Blundell et al. (2015).

    This objective typically adds a KL penalty weighted by a user-defined factor.
    """
    def __init__(self, kl_penalty: float):
        """
        Args:
            kl_penalty (float): The coefficient for scaling KL divergence in the objective.
        """
        self._kl_penalty = kl_penalty

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Combine the loss with scaled KL divergence.

        Args:
            loss (Tensor): Empirical loss (e.g., NLL).
            kl (Tensor): KL divergence between posterior and prior.
            num_samples (float): The number of training samples or an equivalent factor.

        Returns:
            Tensor: A scalar objective = loss + (kl_penalty * KL / num_samples).
        """
        return loss + self._kl_penalty * (kl / num_samples)
