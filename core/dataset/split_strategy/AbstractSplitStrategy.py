from abc import ABC, abstractmethod

from core.dataset.loader import AbstractLoader


class AbstractSplitStrategy(ABC):

    @abstractmethod
    def split(self, dataset_loader: AbstractLoader) -> None:
        # TODO: consider creating Split class and returning split object
        pass
