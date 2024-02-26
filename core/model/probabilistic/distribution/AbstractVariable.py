from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.nn as nn


class AbstractVariable(nn.Module, ABC):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, device: torch.device, fix_mu: bool, fix_rho: bool):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fix_mu)
        self.rho = nn.Parameter(rho, requires_grad=not fix_rho)
        self._device = device
        self.kl_div = None

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log(1 + torch.exp(self.rho))

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_kl(self, other_variable: Self) -> torch.Tensor:
        pass
