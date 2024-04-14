from typing import List, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from core.model import bounded_call


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


def _compute_losses(model: nn.Module,
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


def compute_losses(model: nn.Module,
                   bound_loader: torch.utils.data.DataLoader,
                   mc_samples: int,
                   loss_func_list: List[Callable],
                   device: torch.device,
                   pmin: float = None) -> Tensor:
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
