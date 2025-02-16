from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.utils import KLDivergenceInterface


class AbstractVariable(nn.Module, KLDivergenceInterface, ABC):
    """
    An abstract class representing a single random variable for a probabilistic
    neural network parameter (e.g., weight or bias).

    Each variable holds:
      - A mean parameter (`mu`)
      - A `rho` parameter that is used to derive the standard deviation (`sigma`)
      - A method to sample from the underlying distribution
      - A method to compute KL divergence with another variable of the same type

    This class inherits from `nn.Module` for parameter registration in PyTorch
    and from `KLDivergenceInterface` for consistent KL divergence handling.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        mu_requires_grad: bool = False,
        rho_requires_grad: bool = False,
    ):
        """
        Initialize an AbstractVariable with given `mu` and `rho` tensors.

        Args:
            mu (torch.Tensor): The mean parameter for the distribution.
            rho (torch.Tensor): The parameter from which we derive sigma = log(1 + exp(rho)).
            mu_requires_grad (bool, optional): If True, allow gradients on `mu`.
            rho_requires_grad (bool, optional): If True, allow gradients on `rho`.
        """
        super().__init__()
        self.mu = nn.Parameter(mu.detach().clone(), requires_grad=mu_requires_grad)
        self.rho = nn.Parameter(rho.detach().clone(), requires_grad=rho_requires_grad)
        self.kl_div = None

    @property
    def sigma(self) -> torch.Tensor:
        """
        The standard deviation of the distribution, computed as:
            sigma = log(1 + exp(rho)).

        Returns:
            torch.Tensor: A tensor representing the current standard deviation.
        """
        return torch.log(1 + torch.exp(self.rho))

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Draw a sample from this variable's underlying distribution.

        Returns:
            torch.Tensor: A sampled value of the same shape as `mu`.
        """
        pass

    @abstractmethod
    def compute_kl(self, other: "AbstractVariable") -> torch.Tensor:
        """
        Compute the KL divergence between this variable and another variable
        of the same distribution type.

        Args:
            other (AbstractVariable): Another AbstractVariable instance
                with comparable parameters (e.g., mu, rho).

        Returns:
            torch.Tensor: A scalar tensor with the KL divergence value.
        """
        pass
