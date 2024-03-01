import torch

from core.trainer.objective import AbstractObjective


class BBBObjective(AbstractObjective):
    def __init__(self, kl_penalty: float, pmin: float, num_classes: int, device: torch.device):
        super().__init__(kl_penalty,False, pmin, num_classes, device)

    def bound(self, empirical_risk: float, kl: float, num_samples: float) -> float:
        return empirical_risk + self._kl_penalty * (kl / num_samples)
