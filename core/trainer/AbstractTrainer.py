from abc import ABC, abstractmethod
from typing import Dict

import torch

from core.model import AbstractModel
from core.trainer import TorchOptimizerFactory
from core.trainer.objective import AbstractObjective, ObjectiveFactory


class AbstractTrainer(ABC):
    def __init__(self, device: torch.device):
        self._device = device

    @staticmethod
    def _select_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
        return TorchOptimizerFactory().create(optimizer_name)

    @staticmethod
    def _select_objective(objective_name: str) -> AbstractObjective:
        return ObjectiveFactory().create(objective_name)

    @abstractmethod
    def train(self, *args, **kwargs) -> AbstractModel:
        pass
