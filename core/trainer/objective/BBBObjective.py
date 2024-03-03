import torch
from torch import Tensor

from core.trainer.objective import AbstractObjective


class BBBObjective(AbstractObjective):
    def __init__(
        self, kl_penalty: float, pmin: float, num_classes: int, num_mc_samples: int, delta: float, delta_test: float, device: torch.device
    ):
        super().__init__(kl_penalty, False, pmin, num_classes, num_mc_samples, delta, delta_test, device)

    def bound(self, empirical_risk: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        return empirical_risk + self._kl_penalty * (kl / num_samples)
