from abc import ABC, abstractmethod

import torch


class KLDivergenceInterface(ABC):
    """
    An abstract base class for computing Kullback-Leibler Divergence (KL Divergence).
    """

    @abstractmethod
    def compute_kl(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the Kullback-Leibler Divergence (KL Divergence) between two probability distributions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The computed KL Divergence.
        """
        pass
