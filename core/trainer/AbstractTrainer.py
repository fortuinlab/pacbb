from abc import ABC, abstractmethod
from typing import Dict

from core.model import AbstractModel


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, model: AbstractModel, training_config: Dict) -> AbstractModel:
        pass
