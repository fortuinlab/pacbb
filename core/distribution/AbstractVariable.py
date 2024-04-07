from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.utils import KLDivergenceInterface


class AbstractVariable(nn.Module, KLDivergenceInterface, ABC):
    def __init__(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        mu_requires_grad: bool = False,
        rho_requires_grad: bool = False,
    ):
        super().__init__()
        self.mu = nn.Parameter(mu.detach().clone(), requires_grad=mu_requires_grad)
        self.rho = nn.Parameter(rho.detach().clone(), requires_grad=rho_requires_grad)
        self.kl_div = None

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log(1 + torch.exp(self.rho))

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_kl(self, other: 'AbstractVariable') -> torch.Tensor:
        pass
