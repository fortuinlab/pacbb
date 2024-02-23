from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.nn as nn


class AbstractVariable(nn.Module, ABC):
    def __init__(self, mu, rho, device, fix_mu, fix_rho):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fix_mu)
        # TODO: change to sigma
        self.rho = nn.Parameter(rho, requires_grad=not fix_rho)
        self.device = device

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log(1 + torch.exp(self.rho))

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_kl(self, other_variable: Self) -> torch.Tensor:
        pass
