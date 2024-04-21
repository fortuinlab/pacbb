import numpy as np

import torch
from torch import Tensor
from tqdm import tqdm

from core.distribution import AbstractVariable


class GaussianVariable(AbstractVariable):
    """
    Represents a Gaussian random variable with mean mu and TODO rho.
    """

    def __init__(
        self,
            mu: Tensor,
            rho: Tensor,
            mu_requires_grad:
            bool = False,
            rho_requires_grad: bool = False,
    ):
        """
        Initialize the GaussianVariable.

        Args:
            mu (Tensor): The mean of the Gaussian distribution.
            rho (Tensor): TODO of the Gaussian distribution.
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
        print(kl_div)
        return kl_div

    def density(self, x: Tensor) -> Tensor:
        denominator = self.sigma * np.sqrt(2 * np.pi)
        nominator = torch.exp(-torch.div(torch.square(torch.div((x - self.mu), self.sigma)), 2))
        return nominator / denominator

    def compute_kl_numerical(self, other: "GaussianVariable") -> Tensor:
        mc_samples = 100
        epsilon = torch.finfo(torch.float32).tiny
        epsilon = 1e-8
        expectation = []
        for i in tqdm(range(mc_samples)):
            weight = self.sample()
            x = self.density(weight)
            y = other.density(weight)
            mask = torch.logical_and(x >= epsilon, y >= epsilon)
            ln_term = torch.log(x) - torch.log(y)
            ln_term[~mask] = 0.0
            # ln_term *= self.density(weight)
            expectation.append(ln_term)
        print(torch.stack(expectation).mean(dim=0).sum())
        return torch.stack(expectation).mean(dim=0).sum()
