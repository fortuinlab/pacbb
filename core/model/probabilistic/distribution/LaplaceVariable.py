from typing import Self

import torch
import torch.nn as nn

from core.model.probabilistic.distribution import AbstractVariable


class LaplaceVariable(AbstractVariable):
    def __init__(self, mu, rho, device="cuda", fix_mu=False, fix_rho=False):
        super().__init__(mu, rho, device, fix_mu, fix_rho)

    def sample(self) -> torch.Tensor:
        raise NotImplementedError()

    def compute_kl(self, other: Self) -> torch.Tensor:
        raise NotImplementedError()
