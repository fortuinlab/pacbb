from torch import Tensor

from abc import ABC, abstractmethod


class AbstractObjective(ABC):
    @abstractmethod
    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        pass
