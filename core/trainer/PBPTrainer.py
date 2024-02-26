from typing import Dict

from core.model import AbstractModel
from core.trainer import AbstractTrainer


class PBPTrainer(AbstractTrainer):
    def train(self, model: AbstractModel, training_config: Dict) -> AbstractModel:
        epochs = training_config['epochs']
        disable_tqdm = training_config['disable_tqdm']

        for epoch in trange(epochs, disable=disable_tqdm):

