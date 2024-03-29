from torch import Tensor

from core.objective import AbstractObjective


class BBBObjective(AbstractObjective):
    def __init__(self, kl_penalty: float):
        self._kl_penalty = kl_penalty

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        return loss + self._kl_penalty * (kl / num_samples)
