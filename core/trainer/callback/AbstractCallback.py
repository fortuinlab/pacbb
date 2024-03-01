from abc import ABC, abstractmethod

import torch

from core.model import AbstractModel


class AbstractCallback(ABC):
    def __init__(self, device: torch.device):
        self._device = device

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def finish(self, *args, **kwargs) -> AbstractModel:
        pass
