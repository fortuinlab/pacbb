from typing import Dict, Any
import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from torch.utils.tensorboard import SummaryWriter

from core.distribution import AbstractVariable
from core.distribution.utils import get_params, compute_kl
from core.objective import AbstractObjective
from core.model import bounded_call
from core.layer.utils import get_torch_layers


def train(model: nn.Module,
          posterior: Dict[int, Dict[str, AbstractVariable]],
          prior: Dict[int, Dict[str, AbstractVariable]],
          objective: AbstractObjective,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader,
          parameters: Dict[str, Any],
        ):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=parameters['lr'],
                                momentum=parameters['momentum'])

    if 'seed' in parameters:
        torch.manual_seed(parameters['seed'])
        torch.use_deterministic_algorithms(True)
    for epoch in range(parameters['epochs']):
        for i, (data, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if 'pmin' in parameters:
                output = bounded_call(model, data, parameters['pmin'])
            else:
                output = model(data)
            kl = compute_kl(posterior, prior)
            loss = criterion(output, target)
            objective_value = objective.calculate(loss, kl, parameters['num_samples'])
            objective_value.backward()
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Objective: {objective_value}, Loss: {loss}, KL/n: {kl/parameters['num_samples']}")
    model.train()
