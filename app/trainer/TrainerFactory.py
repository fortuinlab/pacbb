import torch

from app.trainer import (AbstractTrainer, MarglikTrainer,
                         PBPProbabilisticTrainer)
from app.trainer.callback import AbstractCallback
from app.utils import AbstractFactory


class TrainerFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("pbp", PBPProbabilisticTrainer)
        self.register_creator("marglik", MarglikTrainer)

    def create(self, trainer_name: str, callback: AbstractCallback, device: torch.device) -> AbstractTrainer:
        creator = self._creators.get(trainer_name)
        if not creator:
            raise ValueError(f"Invalid trainer: {trainer_name}")
        return creator(callback, device)
