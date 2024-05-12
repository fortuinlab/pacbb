from scripts.utils.dataset.loader import AbstractLoader, MNISTLoader, CIFAR10Loader
from scripts.utils.factory import AbstractFactory


class DataLoaderFactory(AbstractFactory[AbstractLoader]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("mnist", MNISTLoader)
        self.register_creator("cifar10", CIFAR10Loader)
