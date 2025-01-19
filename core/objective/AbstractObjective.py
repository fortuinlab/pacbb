from torch import Tensor

from abc import ABC, abstractmethod


class AbstractObjective(ABC):
    """
    Base class for PAC-Bayes training objectives.

    An objective typically combines:
      - Empirical loss (e.g., negative log-likelihood)
      - KL divergence between posterior and prior
      - Additional terms for confidence or other bounding factors
    """

    @abstractmethod
    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        """
        Compute the combined objective scalar to be backpropagated.

        Args:
            loss (Tensor): Empirical loss, e.g. cross-entropy on a batch.
            kl (Tensor): KL divergence between the current posterior and prior.
            num_samples (float): Number of samples used or total dataset size,
                used for scaling KL or other terms.

        Returns:
            Tensor: A scalar tensor that includes the loss, KL penalty, and any other terms.
        """
        pass
