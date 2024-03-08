from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from app.dataset.split_strategy import PBPSplitStrategy


@dataclass
class FaultySplitStrategy(PBPSplitStrategy):
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

    def _split_not_learnt(
        self,
        train_dataset: data.Dataset,
        test_dataset: data.Dataset,
        split_config: Dict,
        loader_kwargs: Dict,
    ) -> None:
        batch_size = split_config["batch_size"]
        training_percent = split_config["training_percent"]
        val_percent = split_config["val_percent"]

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
        training_percent = split_config["training_percent"]
        val_percent = split_config["val_percent"]
        prior_percent = split_config["prior_percent"]

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
            all_prior_sampler = SubsetRandomSampler(indices_prior)
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
        split_config: Dict,
        loader_kwargs: Dict,
    ) -> None:
        batch_size = split_config["batch_size"]
        training_percent = split_config["training_percent"]
        val_percent = split_config["val_percent"]
        prior_percent = split_config["prior_percent"]

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
            all_prior_sampler = SubsetRandomSampler(indices_prior)
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
