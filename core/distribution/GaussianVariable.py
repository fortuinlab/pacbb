import torch
from torch import Tensor

from core.distribution import AbstractVariable


class GaussianVariable(AbstractVariable):
    """
    Represents a Gaussian random variable with mean mu and rho.
    """

    def __init__(
        self,
        mu: Tensor,
        rho: Tensor,
        mu_requires_grad: bool = False,
        rho_requires_grad: bool = False,
    ):
        """
        Initialize the GaussianVariable.

        Args:
            mu (Tensor): The mean of the Gaussian distribution.
            rho (Tensor): rho = log(exp(sigma)-1) where sigma is a standard deviation of the Gaussian distribution.
            mu_requires_grad (bool): Flag indicating whether mu is fixed.
            rho_requires_grad (bool): Flag indicating whether rho is fixed.
        """
        super().__init__(mu, rho, mu_requires_grad, rho_requires_grad)

    def sample(self) -> Tensor:
        """
        Sample from the Gaussian distribution.

        Returns:
            Tensor: Sampled values from the Gaussian distribution.
        """
        epsilon = torch.randn(self.sigma.size())
        epsilon = epsilon.to(self.mu.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other: "GaussianVariable") -> Tensor:
        """
        Compute the KL divergence between two Gaussian distributions.

        Args:
            other (GaussianVariable): The other Gaussian distribution.

        Returns:
            Tensor: The KL divergence between the two distributions.
        """
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div
