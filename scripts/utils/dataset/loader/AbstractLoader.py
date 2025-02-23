from abc import ABC, abstractmethod

from torch.utils import data


class AbstractLoader(ABC):
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path

    @abstractmethod
    def load(self, seed: int) -> tuple[data.Dataset, data.Dataset]:
        pass
