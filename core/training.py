from typing import Dict, Any, Callable, List

import torch
import torch.nn as nn
from tqdm import tqdm

from core.distribution import AbstractVariable
from core.distribution.utils import get_params, compute_kl
from core.model import bounded_probabilistic_call


def train(model: nn.Module,
          weight_dist: Dict[int, Dict[str, AbstractVariable]],
          prior: Dict[int, Dict[str, AbstractVariable]],
          objective: Any,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader,
          get_layers_func: Callable[[nn.Module], List[nn.Module]],
          parameters: Dict[str, Any],
        ):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': model.parameters()}, *get_params(weight_dist)], parameters['lr'])

    # train
    for epoch in range(parameters['epochs']):
        for data, target in tqdm(train_loader):
            optimizer.zero_grad()
            output = bounded_probabilistic_call(model, parameters['pmin'], data, weight_dist, get_layers_func)
            kl = compute_kl(weight_dist, prior)
            # output = model(data)
            loss = criterion(output, target) + kl * parameters['kl_penalty'] / parameters['num_samples']
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss}")
