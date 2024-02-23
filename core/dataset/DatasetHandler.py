from core.dataset.loader import DatasetLoaderFactory, AbstractLoader
from core.dataset.split_strategy import SplitStrategyFactory, AbstractSplitStrategy


class DatasetHandler:
    """
    Class to handle datasets and split strategies.
    """

    def __init__(self, dataset_name: str, dataset_path: str, split_strategy_name: str):
        """
        Initialize DatasetHandler with dataset and split strategy information.

        Args:
            dataset_name (str): Name of the dataset.
            dataset_path (str): Path to the dataset.
            split_strategy_name (str): Name of the split strategy.
        """
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._split_strategy_name = split_strategy_name
        self._dataset_loader = None
        self._split_strategy = None

    @property
    def dataset_loader(self) -> AbstractLoader:
        """
        Property for accessing the dataset loader.
        """
        return self._dataset_loader

    @property
    def split_strategy(self) -> AbstractSplitStrategy:
        """
        Property for accessing the split strategy.
        """
        return self._split_strategy

    def load_and_split_dataset(self) -> None:
        """
        Load dataset and split it using the specified strategy.
        """
        dataset_loader = DatasetLoaderFactory().create(dataset_name=self._dataset_name, dataset_path=self._dataset_path)
        split_strategy = SplitStrategyFactory().create(split_strategy_name=self._split_strategy_name)

        split_strategy.split(dataset_loader=dataset_loader)

        self._dataset_loader = dataset_loader
        self._split_strategy = split_strategy
