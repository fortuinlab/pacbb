import torch
from torch import Tensor

from core.objective import AbstractObjective
from core.bound import AbstractBound


class BBBObjective(AbstractObjective):
    def __init__(
        self, bound: AbstractBound, kl_penalty: float, pmin: float, num_classes: int, num_mc_samples: int, device: torch.device
    ):
        super().__init__(bound, kl_penalty, False, pmin, num_classes, num_mc_samples, device)

    def bound(self, empirical_risk: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        return empirical_risk + self._kl_penalty * (kl / num_samples)
