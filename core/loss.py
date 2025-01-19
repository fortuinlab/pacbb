from typing import List, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from core.model import bounded_call


def rescale_loss(loss: Tensor, pmin: float) -> Tensor:
    """
    Rescale a loss value by dividing it by log(1/pmin).

    This is often used in PAC-Bayes settings to keep losses within a certain range 
    (e.g., converting losses to the [0, 1] interval) or to improve numerical stability 
    when probabilities must be bounded below by `pmin`.

    Args:
        loss (Tensor): A scalar or batched loss tensor.
        pmin (float): A lower bound for probabilities (e.g., 1e-5).

    Returns:
        Tensor: The loss tensor divided by ln(1/pmin).
    """
    return loss / np.log(1.0 / pmin)


def nll_loss(outputs: Tensor, targets: Tensor, pmin: float = None) -> Tensor:
    """
    Compute the negative log-likelihood (NLL) loss for classification.

    In typical classification settings, `outputs` is the log-probability of each class 
    (e.g., from a log-softmax layer), and `targets` are the true class indices.

    Args:
        outputs (Tensor): Log-probabilities of shape (batch_size, num_classes).
        targets (Tensor): Ground truth class indices of shape (batch_size,).
        pmin (float, optional): Not used directly here; kept for uniform interface with other loss functions.

    Returns:
        Tensor: A scalar tensor representing the average NLL loss over the batch.
    """
    return F.nll_loss(outputs, targets)


def scaled_nll_loss(outputs: Tensor, targets: Tensor, pmin: float) -> Tensor:
    """
    Compute the negative log-likelihood (NLL) loss and then rescale it by log(1/pmin).

    This is a combination of `nll_loss` and `rescale_loss`, often used to ensure 
    that the final loss remains within a desired numeric range for PAC-Bayes optimization.

    Args:
        outputs (Tensor): Log-probabilities of shape (batch_size, num_classes).
        targets (Tensor): Ground truth class indices of shape (batch_size,).
        pmin (float): A lower bound for probabilities.

    Returns:
        Tensor: The rescaled NLL loss as a scalar tensor.
    """
    return rescale_loss(nll_loss(outputs, targets), pmin)


def zero_one_loss(outputs: Tensor, targets: Tensor, pmin: float = None) -> Tensor:
    """
    Compute the 0-1 classification error.

    This function returns a loss between 0 and 1, where 0 indicates perfect 
    classification on the given batch and 1 indicates total misclassification.

    Args:
        outputs (Tensor): Logits or log-probabilities for each class.
        targets (Tensor): Ground truth class indices of shape (batch_size,).
        pmin (float, optional): Not used here; kept for consistency with other losses.

    Returns:
        Tensor: A single-element tensor with the 0-1 error (1 - accuracy).
    """
    predictions = outputs.max(1, keepdim=True)[1]
    correct = predictions.eq(targets.view_as(predictions)).sum().item()
    total = targets.size(0)
    loss_01 = 1 - (correct / total)
    return Tensor([loss_01])


def _compute_losses(model: nn.Module,
                    inputs: Tensor,
                    targets: Tensor,
                    loss_func_list: List[Callable],
                    pmin: float = None) -> List[Tensor]:
    """
    Compute a list of loss values for a single forward pass of the model.

    This function optionally applies a bounded call if `pmin` is specified, then
    evaluates each loss function in `loss_func_list` on the model outputs.

    Args:
        model (nn.Module): A (probabilistic) neural network model.
        inputs (Tensor): Input data for one batch, of shape (batch_size, ...).
        targets (Tensor): Ground truth labels for the batch.
        loss_func_list (List[Callable]): A list of loss functions to compute 
            (e.g., [nll_loss, zero_one_loss]).
        pmin (float, optional): A lower bound for probabilities. If given, 
            `bounded_call` is used before computing losses.

    Returns:
        List[Tensor]: A list of scalar loss tensors, each corresponding to one function 
            in `loss_func_list`.
    """
    if pmin:
        # bound probability to be from [pmin to 1]
        outputs = bounded_call(model, inputs, pmin)
    else:
        outputs = model(inputs)
    losses = []
    for loss_func in loss_func_list:
        loss = loss_func(outputs, targets, pmin) if pmin else loss_func(outputs, targets)
        losses.append(loss)
    return losses


def compute_losses(model: nn.Module,
                   bound_loader: torch.utils.data.DataLoader,
                   mc_samples: int,
                   loss_func_list: List[Callable],
                   device: torch.device,
                   pmin: float = None) -> Tensor:
    """
    Compute average losses over multiple Monte Carlo samples for a given dataset.

    This function is typically used to estimate the expected risk under the 
    posterior by sampling the model `mc_samples` times for each batch in the `bound_loader`.

    Args:
        model (nn.Module): A probabilistic neural network model.
        bound_loader (DataLoader): A DataLoader for the dataset on which 
            the losses should be computed (e.g. a bound or test set).
        mc_samples (int): Number of Monte Carlo samples to draw from the posterior 
            for each batch.
        loss_func_list (List[Callable]): List of loss functions to evaluate 
            (e.g., [nll_loss, zero_one_loss]).
        device (torch.device): The device (CPU/GPU) on which computations are performed.
        pmin (float, optional): A lower bound for probabilities. If provided, 
            `bounded_call` will be used to clamp model outputs.

    Returns:
        Tensor: A tensor of shape (len(loss_func_list),) containing the average losses 
            across the entire dataset for each loss function. The result is typically 
            used to estimate or bound the generalization error in PAC-Bayes experiments.
    """
    with torch.no_grad():
        batch_wise_loss_list = []
        for data, targets in tqdm(bound_loader):
            data, targets = data.to(device), targets.to(device)
            mc_loss_list = []
            for i in range(mc_samples):
                losses = _compute_losses(model, data, targets, loss_func_list, pmin)
                mc_loss_list.append(Tensor(losses))
            batch_wise_loss_list.append(torch.stack(mc_loss_list).mean(dim=0))
    return torch.stack(batch_wise_loss_list).mean(dim=0)
