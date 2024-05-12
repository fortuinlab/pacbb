from abc import ABC, abstractmethod
import math

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


def inv_kl(qs, ks):
    """
    Inversion of the binary KL divergence from (Not) Bounding the True Error by John Langford and Rich Caruana.

    Parameters:
        qs (float): Empirical risk.
        ks (float): Second term for the binary KL divergence inversion.

    Returns:
        float: The computed inversion of the binary KL divergence.
    """
    # TODO: refactor
    ikl = 0
    izq = qs
    dch = 1 - 1e-10
    while True:
        p = (izq + dch) * .5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p) + 0)
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        if (dch - izq) / dch < 1e-5:
            break
    return p
