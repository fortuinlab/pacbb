from typing import Tuple, Union
import math

from torch import Tensor
from core.bound import AbstractBound


class BetterThanKLBound(AbstractBound):
    def __init__(self, delta: float, delta_test: float):
        super().__init__(delta)
        self._delta_test = delta_test

    def calculate(self, avg_loss: float, num_mc_samples: int, kl: Union[Tensor, float], num_samples_bound: int) -> Tuple[Tensor, Tensor]:
        empirical_risk = avg_loss

        risk = math.sqrt(2) * kl + 1 + math.sqrt(math.log((2 * math.sqrt(num_samples_bound)) / self._delta_test))
        risk /= math.sqrt(num_samples_bound)

        return risk, empirical_risk
