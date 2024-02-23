from dataclasses import dataclass
from torch.utils import data
from typing import Union

from core.dataset.split_strategy import AbstractSplitStrategy
from core.dataset.loader import MNISTLoader, CIFAR10Loader


@dataclass
class PBPSplitStrategy(AbstractSplitStrategy):
    posterior_loader: data.dataloader.DataLoader = None
    test_loader: data.dataloader.DataLoader = None
    prior_loader: data.dataloader.DataLoader = None
    bound_loader_1batch: data.dataloader.DataLoader = None
    test_1batch: data.dataloader.DataLoader = None
    bound_loader: data.dataloader.DataLoader = None
    val_loader: data.dataloader.DataLoader = None

    def split(self, dataset_loader: Union[MNISTLoader, CIFAR10Loader]) -> None:
        raise NotImplementedError('Implement logic from loadbatches')
