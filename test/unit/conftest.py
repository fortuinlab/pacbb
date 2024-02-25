import pytest

from core.dataset.loader import DatasetLoaderFactory, AbstractLoader


@pytest.fixture(scope="module")
def dataset_loader(dataset_name: str, dataset_path: str) -> AbstractLoader:
    factory = DatasetLoaderFactory()
    dataset_loader = factory.create(dataset_name, dataset_path)
    return dataset_loader
