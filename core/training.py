from typing import Dict, Any
import logging
import wandb

import ivon
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from core.distribution.utils import compute_kl, DistributionT
from core.objective import AbstractObjective
from core.model import bounded_call


def __raise_exception_on_invalid_value(value: torch.Tensor):
    """
    Raise a ValueError if the given tensor is None or contains NaN values.

    This helper function is used to catch numerical issues that might arise
    during training (e.g., invalid gradients, extreme KL values).

    Args:
        value (torch.Tensor): A scalar tensor to check for validity.

    Raises:
        ValueError: If `value` is None or if any entry is NaN.
    """
    if value is None or torch.isnan(value).any():
        raise ValueError(f"Invalid value {value}")


def train(model: nn.Module,
          posterior: DistributionT,
          prior: DistributionT,
          objective: AbstractObjective,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader,
          parameters: Dict[str, Any],
          device: torch.device,
          wandb_params: Dict = None,
        ):
    """
    Train a probabilistic neural network by optimizing a PAC-Bayes-inspired objective.

    At each iteration:
      1) Optionally clamp model outputs using `bounded_call` if `pmin` is provided in `parameters`.
      2) Compute KL divergence between posterior and prior.
      3) Compute the empirical loss (NLL by default).
      4) Combine loss and KL via the given `objective`.
      5) Backpropagate and update model parameters.

    Logs intermediate results (objective, loss, KL) to Python's logger and optionally to wandb.

    Args:
        model (nn.Module): The probabilistic neural network to train.
        posterior (DistributionT): The current (learnable) posterior distribution.
        prior (DistributionT): The (fixed or partially learnable) prior distribution.
        objective (AbstractObjective): An object that merges empirical loss and KL 
            into a single differentiable objective.
        train_loader (DataLoader): Dataloader for the training dataset.
        val_loader (DataLoader): Dataloader for the validation dataset (currently unused here).
        parameters (Dict[str, Any]): A dictionary of training hyperparameters, which can include:
            - 'lr': Learning rate.
            - 'momentum': Momentum term for SGD.
            - 'epochs': Number of epochs.
            - 'num_samples': Usually the size of the training set (or mini-batch size times steps).
            - 'seed': Random seed (optional).
            - 'pmin': Minimum probability for bounding (optional).
        device (torch.device): The device (CPU or GPU) for training.
        wandb_params (Dict, optional): Configuration for logging to Weights & Biases. Expects keys:
            - "log_wandb": bool, whether to log or not
            - "name_wandb": str, run name / prefix for logging

    Returns:
        None: The model (and its posterior) are updated in-place over the specified epochs.
    """
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=parameters['lr'],
                                momentum=parameters['momentum'])

    if 'seed' in parameters:
        torch.manual_seed(parameters['seed'])
    for epoch in range(parameters['epochs']):
        for i, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if 'pmin' in parameters:
                output = bounded_call(model, data, parameters['pmin'])
            else:
                output = model(data)
            kl = compute_kl(posterior, prior)
            loss = criterion(output, target)
            objective_value = objective.calculate(loss, kl, parameters['num_samples'])
            __raise_exception_on_invalid_value(objective_value)
            objective_value.backward()
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Objective: {objective_value}, Loss: {loss}, KL/n: {kl/parameters['num_samples']}")
        if wandb_params is not None and wandb_params["log_wandb"]:
            wandb.log({wandb_params["name_wandb"] + '/Epoch': epoch,
                       wandb_params["name_wandb"] + '/Objective': objective_value,
                       wandb_params["name_wandb"] + '/Loss': loss,
                       wandb_params["name_wandb"] + '/KL-n': kl/parameters['num_samples']})
