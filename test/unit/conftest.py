import pytest
import torch

from app.dataset.loader import AbstractLoader, DatasetLoaderFactory
from app.model import PBP3Model


@pytest.fixture(scope="module")
def dataset_loader(dataset_name: str, dataset_path: str) -> AbstractLoader:
    factory = DatasetLoaderFactory()
    dataset_loader = factory.create(dataset_name, dataset_path)
    return dataset_loader


@pytest.fixture(scope="module")
def pbp3_model() -> PBP3Model:
    return PBP3Model(28 * 28, 100, 10, "gaussian", 0.01, "zeros", torch.device("cpu"))
