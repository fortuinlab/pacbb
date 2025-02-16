from dataclasses import dataclass

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from core.split_strategy import PBPSplitStrategy


@dataclass
class FaultySplitStrategy(PBPSplitStrategy):
    """
    A specialized (and potentially buggy) subclass of PBPSplitStrategy that demonstrates
    alternative splitting logic or partial overlaps between dataset subsets.

    Fields:
        posterior_loader (DataLoader): DataLoader for posterior training.
        prior_loader (DataLoader): DataLoader for prior training.
        val_loader (DataLoader): DataLoader for validation set.
        test_loader (DataLoader): DataLoader for test set.
        test_1batch (DataLoader): DataLoader for test set (one big batch).
        bound_loader (DataLoader): DataLoader for bound evaluation set.
        bound_loader_1batch (DataLoader): DataLoader for bound evaluation set (one big batch).
    """

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

    def __init__(
        self,
        prior_type: str,
        train_percent: float,
        val_percent: float,
        prior_percent: float,
        self_certified: bool,
    ):
        """
        Initialize the FaultySplitStrategy with user-defined percentages
        and flags for how to partition the dataset.

        Args:
            prior_type (str): A string indicating how the prior is handled (e.g. "not_learnt", "learnt").
            train_percent (float): Fraction of dataset to use for training.
            val_percent (float): Fraction of dataset to use for validation.
            prior_percent (float): Fraction of dataset to use for prior training.
            self_certified (bool): If True, indicates self-certified approach to data splitting.
        """
        super().__init__(
            prior_type, train_percent, val_percent, prior_percent, self_certified
        )

    def _split_not_learnt(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: dict,
        loader_kwargs: dict,
    ) -> None:
        """
        Split the data for the case when the prior is not learned from data.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The test dataset.
            split_config (Dict): A configuration dictionary containing keys like 'batch_size', 'seed', etc.
            loader_kwargs (Dict): Additional keyword arguments for DataLoader (e.g., num_workers).
        """
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent

        train_size = len(train_dataset.data)
        test_size = len(test_dataset.data)

        indices = list(range(train_size))
        split = int(np.round((training_percent) * train_size))
        np.random.seed(split_config["seed"])
        np.random.shuffle(indices)

        if val_percent > 0.0:
            # compute number of data points
            indices = list(range(split))
            split_val = int(np.round((val_percent) * split))
            train_idx, val_idx = indices[split_val:], indices[:split_val]
        else:
            train_idx = indices[:split]
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
        split_config: dict,
        loader_kwargs: dict,
    ) -> None:
        """
        Split logic when the prior is learned from data and we use a self-certified approach.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The test dataset.
            split_config (Dict): A dictionary of split settings (batch size, seed, etc.).
            loader_kwargs (Dict): Keyword arguments for DataLoader initialization.
        """
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent
        prior_percent = self._prior_percent

        n = len(train_dataset.data) + len(test_dataset.data)

        # reduce training data if needed
        new_num_train = int(np.round((training_percent) * n))
        indices = list(range(new_num_train))
        split = int(np.round((prior_percent) * new_num_train))
        np.random.seed(split_config["seed"])
        np.random.shuffle(indices)

        all_train_sampler = SubsetRandomSampler(indices)
        if val_percent > 0.0:
            bound_idx = indices[split:]
            indices_prior = list(range(split))
            _all_prior_sampler = SubsetRandomSampler(indices_prior)
            split_val = int(np.round((val_percent) * split))
            prior_idx, val_idx = indices_prior[split_val:], indices_prior[:split_val]
        else:
            bound_idx, prior_idx = indices[split:], indices[:split]
            val_idx = None

        bound_sampler = SubsetRandomSampler(bound_idx)
        prior_sampler = SubsetRandomSampler(prior_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        final_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

        self.posterior_loader = torch.utils.data.DataLoader(
            final_dataset,
            batch_size=batch_size,
            sampler=all_train_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        self.prior_loader = torch.utils.data.DataLoader(
            final_dataset,
            batch_size=batch_size,
            sampler=prior_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        if val_idx:
            self.val_loader = torch.utils.data.DataLoader(
                final_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        # self.test_loader = None
        # self.test_1batch = None
        self.bound_loader = torch.utils.data.DataLoader(
            final_dataset,
            batch_size=batch_size,
            sampler=bound_sampler,
            shuffle=False,
            **loader_kwargs,
        )
        self.bound_loader_1batch = torch.utils.data.DataLoader(
            final_dataset,
            batch_size=len(bound_idx),
            sampler=bound_sampler,
            **loader_kwargs,
        )

    def _split_learnt_not_self_certified(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: dict,
        loader_kwargs: dict,
    ) -> None:
        """
        Split logic for a learned prior without self-certification.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The test dataset.
            split_config (Dict): Dictionary with split hyperparameters (batch size, seed, etc.).
            loader_kwargs (Dict): Additional arguments for torch.utils.data.DataLoader.
        """
        batch_size = split_config["batch_size"]
        training_percent = self._train_percent
        val_percent = self._val_percent
        prior_percent = self._prior_percent

        train_size = len(train_dataset.data)
        test_size = len(test_dataset.data)

        new_num_train = int(np.round((training_percent) * train_size))
        indices = list(range(new_num_train))
        split = int(np.round((prior_percent) * new_num_train))
        np.random.seed(split_config["seed"])
        np.random.shuffle(indices)

        all_train_sampler = SubsetRandomSampler(indices)
        # train_idx, valid_idx = indices[split:], indices[:split]
        if val_percent > 0.0:
            bound_idx = indices[split:]
            indices_prior = list(range(split))
            _all_prior_sampler = SubsetRandomSampler(indices_prior)
            split_val = int(np.round((val_percent) * split))
            prior_idx, val_idx = indices_prior[split_val:], indices_prior[:split_val]
        else:
            bound_idx, prior_idx = indices[split:], indices[:split]
            val_idx = None

        bound_sampler = SubsetRandomSampler(bound_idx)
        prior_sampler = SubsetRandomSampler(prior_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.posterior_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=all_train_sampler,
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
