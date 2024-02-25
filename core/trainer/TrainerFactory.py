from core.trainer import AbstractTrainer, MarglikTrainer, PBPTrainer
from core.utils import AbstractFactory


class TrainerFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("pbp", PBPTrainer)
        self.register_creator("marglik", MarglikTrainer)

    def create(self, trainer_name: str) -> AbstractTrainer:
        creator = self._creators.get(trainer_name)
        if not creator:
            raise ValueError(f"Invalid trainer: {trainer_name}")
        return creator(trainer_name)
