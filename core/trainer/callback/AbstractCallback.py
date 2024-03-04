from abc import ABC, abstractmethod

import torch
from torch.utils import data

from core.model import AbstractModel
from core.model.probabilistic import AbstractPBPModel
from core.trainer.objective import AbstractObjective


class AbstractCallback(ABC):
    def __init__(self, device: torch.device):
        self._device = device

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def process(self,
                epoch: int,
                model: AbstractPBPModel,
                train_loader: data.Dataset,
                val_loader: data.Dataset,
                objective: AbstractObjective) -> None:
        pass

    @abstractmethod
    def finish(self,
               model: AbstractPBPModel,
               train_loader: data.Dataset,
               val_loader: data.Dataset,
               objective: AbstractObjective) -> AbstractModel:
        pass
