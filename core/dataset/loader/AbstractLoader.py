from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils import data


class AbstractLoader(ABC):
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path

    @abstractmethod
    def load(self, seed: int) -> Tuple[data.Dataset, data.Dataset]:
        pass
