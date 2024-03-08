from app.dataset.loader import AbstractLoader, CIFAR10Loader, MNISTLoader
from app.utils import AbstractFactory


class DatasetLoaderFactory(AbstractFactory):
    """
    Factory class for creating dataset loader objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("mnist", MNISTLoader)
        self.register_creator("cifar10", CIFAR10Loader)

    def create(self, dataset_name: str, dataset_path: str) -> AbstractLoader:
        """
        Create a dataset loader object.

        Args:
            dataset_name (str): Name of the dataset.
            dataset_path (str): Path to the dataset.

        Returns:
            AbstractLoader: Created dataset loader object.

        Raises:
            ValueError: If an invalid dataset name is provided.
        """
        creator = self._creators.get(dataset_name)
        if not creator:
            raise ValueError(f"Invalid dataset: {dataset_name}")
        return creator(dataset_path)
