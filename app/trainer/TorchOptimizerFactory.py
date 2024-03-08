import torch.optim as optim

from app.utils import AbstractFactory


class TorchOptimizerFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("sgd", optim.SGD)

    def create(self, optimizer_name: str, *args, **kwargs) -> optim.Optimizer:
        creator = self._creators.get(optimizer_name)
        if not creator:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")
        return creator(*args, **kwargs)
