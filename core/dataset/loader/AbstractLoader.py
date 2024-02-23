from abc import ABC, abstractmethod
from torch.utils import data
from typing import Tuple


class AbstractLoader(ABC):
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path

    @staticmethod
    @abstractmethod
    def load() -> Tuple[data.Dataset, data.Dataset]:
        pass
