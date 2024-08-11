from torch import Tensor
from typing import List

from abc import ABC, abstractmethod


class AbstractSamplingObjective(ABC):
    def __init__(self, kl_penalty: float, n: int) -> None:
        self._kl_penalty = kl_penalty
        self.n: int = n

    @abstractmethod
    def calculate(self, losses: List[Tensor], kl: Tensor, num_samples: float) -> Tensor:
        pass
