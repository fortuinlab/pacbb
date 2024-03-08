from typing import Dict

from app.model import AbstractModel
from app.trainer import AbstractTrainer


class MarglikTrainer(AbstractTrainer):
    def train(self, model: AbstractModel, training_config: Dict) -> AbstractModel:
        pass
