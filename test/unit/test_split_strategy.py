from typing import Any, List, Set

import pytest

from app.dataset.loader import DatasetLoaderFactory
from app.dataset.split_strategy import PBPSplitStrategy


# TODO: move to utils
def get_intersections(*lists: List[Any]) -> Set[Any]:
    sets = [set(list_) for list_ in lists]
    intersection = set.intersection(*sets)
    return intersection


@pytest.mark.parametrize(
    "dataset_name, dataset_path, expected_lengths",
    [
        # (dataset_name, dataset_path, [posterior, prior, val, test, test_1batch, bound, bound_1batch])
        ("mnist", "./data/test/dataset/mnist/", (228, None, 12, 40, 1, 228, 1)),
        ("cifar10", "./data/test/dataset/cifar10/", (190, None, 10, 40, 1, 190, 1)),
    ],
)
def test_pbp_split_strategy_not_learnt_prior(
    dataset_name, dataset_path, expected_lengths
):
    factory = DatasetLoaderFactory()
    dataset_loader = factory.create(dataset_name, dataset_path)

    strategy = PBPSplitStrategy()

    split_config = {
        "batch_size": 250,
        "training_percent": 1.0,
        "val_percent": 0.05,
        "prior_percent": 0.5,
        "prior_type": "not_learnt",
        "self_certified": None,
        "seed": 10,
        "dataset_loader_seed": 7,
    }

    strategy.split(dataset_loader=dataset_loader, split_config=split_config)

    assert strategy.posterior_loader is not None
    assert strategy.prior_loader is None
    assert strategy.val_loader is not None
    assert strategy.test_loader is not None
    assert strategy.test_1batch is not None
    assert strategy.bound_loader is not None
    assert strategy.bound_loader_1batch is not None

    intersections = get_intersections(
        strategy.val_loader.sampler.indices, strategy.bound_loader.sampler.indices
    )
    assert len(intersections) == 0

    assert len(strategy.posterior_loader) == expected_lengths[0]
    # assert strategy.prior_loader is None
    assert len(strategy.val_loader) == expected_lengths[2]
    assert len(strategy.test_loader) == expected_lengths[3]
    assert len(strategy.test_1batch) == expected_lengths[4]
    assert len(strategy.bound_loader) == expected_lengths[5]
    assert len(strategy.bound_loader_1batch) == expected_lengths[6]


@pytest.mark.parametrize(
    "dataset_name, dataset_path, expected_lengths",
    [
        # (dataset_name, dataset_path, [posterior, prior, val, test, test_1batch, bound, bound_1batch])
        ("mnist", "./data/test/dataset/mnist/", (280, 133, 7, None, None, 140, 1)),
        ("cifar10", "./data/test/dataset/cifar10/", (240, 114, 6, None, None, 120, 1)),
    ],
)
def test_pbp_split_strategy_learnt_self_certified_prior(
    dataset_name, dataset_path, expected_lengths
):
    factory = DatasetLoaderFactory()
    dataset_loader = factory.create(dataset_name, dataset_path)

    strategy = PBPSplitStrategy()

    split_config = {
        "batch_size": 250,
        "training_percent": 1.0,
        "val_percent": 0.05,
        "prior_percent": 0.5,
        "prior_type": "learnt",
        "self_certified": True,
        "seed": 10,
        "dataset_loader_seed": 7,
    }

    strategy.split(dataset_loader=dataset_loader, split_config=split_config)

    assert strategy.posterior_loader is not None
    assert strategy.prior_loader is not None
    assert strategy.val_loader is not None
    assert strategy.test_loader is None
    assert strategy.test_1batch is None
    assert strategy.bound_loader is not None
    assert strategy.bound_loader_1batch is not None

    intersections = get_intersections(
        strategy.val_loader.sampler.indices,
        strategy.bound_loader.sampler.indices,
        strategy.bound_loader.sampler.indices,
    )

    assert len(intersections) == 0

    assert len(strategy.posterior_loader) == expected_lengths[0]
    assert len(strategy.prior_loader) == expected_lengths[1]
    assert len(strategy.val_loader) == expected_lengths[2]
    # assert strategy.test_loader is None
    # assert strategy.test_1batch is None
    assert len(strategy.bound_loader) == expected_lengths[5]
    assert len(strategy.bound_loader_1batch) == expected_lengths[6]


@pytest.mark.parametrize(
    "dataset_name, dataset_path, expected_lengths",
    [
        # (dataset_name, dataset_path, [posterior, prior, val, test, test_1batch, bound, bound_1batch])
        ("mnist", "./data/test/dataset/mnist/", (240, 114, 6, 40, 1, 120, 1)),
        ("cifar10", "./data/test/dataset/cifar10/", (200, 95, 5, 40, 1, 100, 1)),
    ],
)
def test_pbp_split_strategy_learnt_not_self_certified_prior(
    dataset_name, dataset_path, expected_lengths
):
    factory = DatasetLoaderFactory()
    dataset_loader = factory.create(dataset_name, dataset_path)

    strategy = PBPSplitStrategy()

    split_config = {
        "batch_size": 250,
        "training_percent": 1.0,
        "val_percent": 0.05,
        "prior_percent": 0.5,
        "prior_type": "learnt",
        "self_certified": False,
        "seed": 10,
        "dataset_loader_seed": 7,
    }

    strategy.split(dataset_loader=dataset_loader, split_config=split_config)

    assert strategy.posterior_loader is not None
    assert strategy.prior_loader is not None
    assert strategy.val_loader is not None
    assert strategy.test_loader is not None
    assert strategy.test_1batch is not None
    assert strategy.bound_loader is not None
    assert strategy.bound_loader_1batch is not None

    intersections = get_intersections(
        strategy.val_loader.sampler.indices,
        strategy.bound_loader.sampler.indices,
        strategy.bound_loader.sampler.indices,
    )

    assert len(intersections) == 0

    assert len(strategy.posterior_loader) == expected_lengths[0]
    assert len(strategy.prior_loader) == expected_lengths[1]
    assert len(strategy.val_loader) == expected_lengths[2]
    assert len(strategy.test_loader) == expected_lengths[3]
    assert len(strategy.test_1batch) == expected_lengths[4]
    assert len(strategy.bound_loader) == expected_lengths[5]
    assert len(strategy.bound_loader_1batch) == expected_lengths[6]
