from abc import ABC, abstractmethod
from typing import Dict

from scripts.utils.dataset.loader import AbstractLoader


class AbstractSplitStrategy(ABC):
    """
    An abstract interface for splitting a dataset loader into separate subsets
    for training, prior training, validation, testing, or bound evaluation
    in a PAC-Bayes pipeline.

    Different implementations can define how the data is partitioned,
    ensuring that prior data, posterior data, and bound data do not overlap.
    """
    @abstractmethod
    def split(self, dataset_loader: AbstractLoader, split_config: Dict) -> None:
        """
        Partition the data from `dataset_loader` according to the configuration in `split_config`.

        Args:
            dataset_loader (AbstractLoader): A loader or dataset manager providing the raw dataset.
            split_config (Dict): A dictionary specifying how to split the data 
                (e.g., batch_size, train/val/test percentages, random seeds, etc.).

        Returns:
            None: The method typically sets up instance attributes such as
            `posterior_loader`, `prior_loader`, etc. for later access.
        """
        pass
