from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from core.dataset.loader import CIFAR10Loader, MNISTLoader
from core.split_strategy import AbstractSplitStrategy


@dataclass
class PBPSplitStrategy(AbstractSplitStrategy):
    # Posterior training
    posterior_loader: data.dataloader.DataLoader = None
    # Prior training
    prior_loader: data.dataloader.DataLoader = None
    # Evaluation
    val_loader: data.dataloader.DataLoader = None
    test_loader: data.dataloader.DataLoader = None
    test_1batch: data.dataloader.DataLoader = None
    # Bounds evaluation
    bound_loader: data.dataloader.DataLoader = None
    bound_loader_1batch: data.dataloader.DataLoader = None

    def __init__(self, prior_type: str, train_percent: float, val_percent: float, prior_percent: float, self_certified: bool):
        self._prior_type = prior_type
        self._train_percent = train_percent
        self._val_percent = val_percent
        self._prior_percent = prior_percent
        self._self_certified = self_certified

    def _split_not_learnt(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: Dict,
        loader_kwargs: Dict,
    ) -> None:
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent
        seed = split_config["seed"]

        train_size = len(train_dataset.data)
        test_size = len(test_dataset.data)
        train_indices = list(range(train_size))
        np.random.seed(seed)
        np.random.shuffle(train_indices)

        # take fraction of a training dataset
        training_split = int(np.round(training_percent * train_size))
        train_indices = train_indices[:training_split]
        # TODO: Could be a bug here because of logic change
        if val_percent > 0.0:
            val_split = int(np.round(val_percent * training_split))
            train_idx = train_indices[val_split:]
            val_idx = train_indices[:val_split]
        else:
            train_idx = train_indices
            val_idx = None

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.posterior_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            **loader_kwargs,
        )
        # self.prior_loader = None
        if val_idx:
            self.val_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
            )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
        )
        self.test_1batch = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_size, shuffle=True, **loader_kwargs
        )
        # TODO: can be a bug here because it was =posterior_loader
        self.bound_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            **loader_kwargs,
        )
        self.bound_loader_1batch = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_idx),
            sampler=train_sampler,
            **loader_kwargs,
        )

    def _split_learnt_self_certified(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: Dict,
        loader_kwargs: Dict,
    ) -> None:
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent
        prior_percent = self._prior_percent
        seed = split_config["seed"]

        train_test_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset]
        )
        train_test_size = len(train_dataset.data) + len(test_dataset.data)
        train_indices = list(range(train_test_size))
        np.random.seed(seed)
        np.random.shuffle(train_indices)
        # take fraction of a training dataset
        training_test_split = int(np.round(training_percent * train_test_size))
        train_indices = train_indices[:training_test_split]

        if val_percent > 0.0:
            prior_split = int(np.round(prior_percent * training_test_split))
            bound_idx, prior_val_idx = (
                train_indices[prior_split:],
                train_indices[:prior_split],
            )
            val_split = int(np.round(val_percent * prior_split))
            prior_idx, val_idx = (
                prior_val_idx[val_split:],
                prior_val_idx[:val_split],
            )
        else:
            prior_split = int(np.round(prior_percent * training_test_split))
            bound_idx, prior_idx = (
                train_indices[prior_split:],
                train_indices[:prior_split],
            )
            val_idx = None

        train_test_sampler = SubsetRandomSampler(train_indices)
        bound_sampler = SubsetRandomSampler(bound_idx)
        prior_sampler = SubsetRandomSampler(prior_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.posterior_loader = torch.utils.data.DataLoader(
            train_test_dataset,
            batch_size=batch_size,
            sampler=train_test_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        self.prior_loader = torch.utils.data.DataLoader(
            train_test_dataset,
            batch_size=batch_size,
            sampler=prior_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        if val_idx:
            self.val_loader = torch.utils.data.DataLoader(
                train_test_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        # self.test_loader = None
        # self.test_1batch = None
        self.bound_loader = torch.utils.data.DataLoader(
            train_test_dataset,
            batch_size=batch_size,
            sampler=bound_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        self.bound_loader_1batch = torch.utils.data.DataLoader(
            train_test_dataset,
            batch_size=len(bound_idx),
            sampler=bound_sampler,
            **loader_kwargs,
        )

    def _split_learnt_not_self_certified(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: Dict,
        loader_kwargs: Dict,
    ) -> None:
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent
        prior_percent = self._prior_percent
        seed = split_config["seed"]

        train_size = len(train_dataset.data)
        test_size = len(test_dataset.data)
        train_indices = list(range(train_size))
        # TODO: no need to shuffle because of SubsetRandomSampler
        np.random.seed(seed)
        np.random.shuffle(train_indices)

        training_split = int(np.round(training_percent * train_size))
        train_indices = train_indices[:training_split]

        if val_percent > 0.0:
            prior_split = int(np.round(prior_percent * training_split))
            bound_idx, prior_val_idx = (
                train_indices[prior_split:],
                train_indices[:prior_split],
            )
            val_split = int(np.round(val_percent * prior_split))
            prior_idx, val_idx = (
                prior_val_idx[val_split:],
                prior_val_idx[:val_split],
            )
        else:
            prior_split = int(np.round(prior_percent * training_split))
            bound_idx, prior_idx = (
                train_indices[prior_split:],
                train_indices[:prior_split],
            )
            val_idx = None

        train_sampler = SubsetRandomSampler(train_indices)
        bound_sampler = SubsetRandomSampler(bound_idx)
        prior_sampler = SubsetRandomSampler(prior_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.posterior_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        self.prior_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=prior_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        if val_idx:
            self.val_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
        )
        self.test_1batch = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_size, shuffle=True, **loader_kwargs
        )
        self.bound_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=bound_sampler,
            shuffle=False,
        )
        self.bound_loader_1batch = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(bound_idx),
            sampler=bound_sampler,
            **loader_kwargs,
        )

    def split(
        self, dataset_loader: Union[MNISTLoader, CIFAR10Loader], split_config: Dict
    ) -> None:
        dataset_loader_seed = split_config["dataset_loader_seed"]
        train_dataset, test_dataset = dataset_loader.load(dataset_loader_seed)

        loader_kwargs = (
            {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
        )

        if self._prior_type == "not_learnt":
            self._split_not_learnt(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                split_config=split_config,
                loader_kwargs=loader_kwargs,
            )
        elif self._prior_type == "learnt":
            if self._self_certified:
                self._split_learnt_self_certified(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    split_config=split_config,
                    loader_kwargs=loader_kwargs,
                )
            else:
                self._split_learnt_not_self_certified(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    split_config=split_config,
                    loader_kwargs=loader_kwargs,
                )
        else:
            raise ValueError(f"Invalid prior_type: {self._prior_type}")
