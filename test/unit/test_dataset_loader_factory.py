import pytest
from torch.utils import data

from core.dataset.loader import DatasetLoaderFactory


@pytest.mark.parametrize(
    "dataset_name, dataset_path",
    [
        ("mnist", "./data/test/dataset/mnist/"),
        ("cifar10", "./data/test/dataset/cifar10/"),
    ],
)
def test_dataset_loader(dataset_name, dataset_path):
    factory = DatasetLoaderFactory()
    dataset = factory.create(dataset_name, dataset_path)
    train, test = dataset.load()
    assert isinstance(train, data.Dataset)
    assert isinstance(test, data.Dataset)
    assert len(train) > 0
    assert len(test) > 0

