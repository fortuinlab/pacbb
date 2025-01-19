import torch
from torch import Tensor

from core.distribution import AbstractVariable


class LaplaceVariable(AbstractVariable):
    def __init__(
        self,
            mu: Tensor,
            rho: Tensor,
            mu_requires_grad:
            bool = False,
            rho_requires_grad: bool = False,
    ):
        """
        Initialize the LaplaceVariable.

        Args:
            mu (Tensor): The mean of the Laplace distribution.
            rho (Tensor): rho = log(exp(sigma)-1) of the Laplace distribution.
            mu_requires_grad (bool): Flag indicating whether mu is fixed.
            rho_requires_grad (bool): Flag indicating whether rho is fixed.
        """
        super().__init__(mu, rho, mu_requires_grad, rho_requires_grad)

    def sample(self):
        """
        Sample from the Laplace distribution.

        Returns:
            Tensor: Sampled values from the Laplace distribution.
        """
        epsilon = 0.999 * torch.rand(self.sigma.size()) - 0.49999
        epsilon = epsilon.to(self.mu.device)
        return self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)), torch.log(1-2*torch.abs(epsilon)))

    def compute_kl(self, other: "LaplaceVariable") -> Tensor:
        """
        Compute the KL divergence between two Laplace distributions.

        Args:
            other (LaplaceVariable): The other Laplace distribution.

        Returns:
            Tensor: The KL divergence between the two distributions.
        """
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div
