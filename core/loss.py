from typing import List, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from core.model import bounded_call
from core.distribution import AbstractVariable


def rescale_loss(loss: Tensor, pmin: float) -> Tensor:
    return loss / np.log(1.0 / pmin)


def nll_loss(outputs: Tensor, targets: Tensor, pmin: float = None) -> Tensor:
    return F.nll_loss(outputs, targets)


def scaled_nll_loss(outputs: Tensor, targets: Tensor, pmin: float) -> Tensor:
    return rescale_loss(nll_loss(outputs, targets), pmin)


def zero_one_loss(outputs: Tensor, targets: Tensor, pmin: float = None) -> Tensor:
    predictions = outputs.max(1, keepdim=True)[1]
    correct = predictions.eq(targets.view_as(predictions)).sum().item()
    total = targets.size(0)
    loss_01 = 1 - (correct / total)
    return Tensor([loss_01])


def compute_losses(model: nn.Module,
                   inputs: Tensor,
                   targets: Tensor,
                   loss_func_list: List[Callable],
                   pmin: float = None) -> List[Tensor]:
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


def compute_avg_losses(model: nn.Module,
                       inputs: Tensor,
                       targets: Tensor,
                       mc_samples: int,
                       loss_func_list: List[Callable],
                       pmin: float = None) -> Tensor:
    losses_list = []
    for i in range(mc_samples):
        losses_list.append(compute_losses(model,
                                          inputs,
                                          targets,
                                          loss_func_list,
                                          pmin))
    return torch.Tensor(losses_list).mean(dim=0)
