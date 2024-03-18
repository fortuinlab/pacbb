from abc import ABC, abstractmethod
from typing import Dict

from core.dataset.loader import AbstractLoader


class AbstractSplitStrategy(ABC):

    @abstractmethod
    def split(self, dataset_loader: AbstractLoader, split_config: Dict) -> None:
        # TODO: consider creating Split class and returning split object
        pass
