from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor


class AbstractBound(ABC):
    def __init__(self, delta: float):
        self._delta = delta

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Union[Tensor, Tensor]:
        pass
