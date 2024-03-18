from typing import Tuple

import torch
from torch.utils import data
from torchvision import datasets, transforms

from core.dataset.loader import AbstractLoader


class CIFAR10Loader(AbstractLoader):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def load(self, seed: int = 7) -> Tuple[data.Dataset, data.Dataset]:
        torch.manual_seed(seed)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train = datasets.CIFAR10(
            self._dataset_path, train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            self._dataset_path, train=False, download=True, transform=transform
        )
        return train, test
