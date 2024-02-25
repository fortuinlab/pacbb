from typing import Dict

from core.model import AbstractModel
from core.trainer import AbstractTrainer


class PBPTrainer(AbstractTrainer):
    def train(self, model: AbstractModel, training_config: Dict) -> AbstractModel:
        pass
