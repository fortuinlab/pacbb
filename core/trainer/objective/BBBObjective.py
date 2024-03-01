import torch
from torch import Tensor

from core.trainer.objective import AbstractObjective


class BBBObjective(AbstractObjective):
    def __init__(
        self, kl_penalty: float, pmin: float, num_classes: int, device: torch.device
    ):
        super().__init__(kl_penalty, False, pmin, num_classes, device)

    def bound(self, empirical_risk: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        return empirical_risk + self._kl_penalty * (kl / num_samples)
