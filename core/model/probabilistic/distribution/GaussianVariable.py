import torch
from torch import Tensor

from core.model.probabilistic.distribution import AbstractVariable


class GaussianVariable(AbstractVariable):
    """
    Represents a Gaussian random variable with mean mu and TODO rho.
    """

    def __init__(
        self, mu: Tensor, rho: Tensor, device: torch.device, fix_mu: bool, fix_rho: bool
    ):
        """
        Initialize the GaussianVariable.

        Args:
            mu (Tensor): The mean of the Gaussian distribution.
            rho (Tensor): TODO of the Gaussian distribution.
            device (torch.device): The device to which tensors should be moved.
            fix_mu (bool): Flag indicating whether mu is fixed.
            fix_rho (bool): Flag indicating whether rho is fixed.
        """
        super().__init__(mu, rho, device, fix_mu, fix_rho)

    def sample(self) -> Tensor:
        """
        Sample from the Gaussian distribution.

        Returns:
            Tensor: Sampled values from the Gaussian distribution.
        """
        epsilon = torch.randn_like(self.sigma, device=self._device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other: "GaussianVariable") -> Tensor:
        """
        Compute the KL divergence between two Gaussian distributions.

        Args:
            other (GaussianVariable): The other Gaussian distribution.

        Returns:
            Tensor: The KL divergence between the two distributions.
        """
        # TODO: 'GaussianVariable' or Self?
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div
