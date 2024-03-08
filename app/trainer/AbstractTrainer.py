from abc import ABC, abstractmethod

import torch

from app.model import AbstractModel
from app.trainer import TorchOptimizerFactory
from app.objective import ObjectiveFactory
from core.objective import AbstractObjective


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
