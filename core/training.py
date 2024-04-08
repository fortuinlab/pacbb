from typing import Dict, Any
import logging

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from core.distribution.utils import compute_kl, DistributionT
from core.objective import AbstractObjective
from core.model import bounded_call


def train(model: nn.Module,
          posterior: DistributionT,
          prior: DistributionT,
          objective: AbstractObjective,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader,
          parameters: Dict[str, Any],
          device: torch.device,
        ):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=parameters['lr'],
                                momentum=parameters['momentum'])

    if 'seed' in parameters:
        torch.manual_seed(parameters['seed'])
    for epoch in range(parameters['epochs']):
        for i, (data, target) in tqdm(enumerate(train_loader)):
            data, targets = data.to(device), target.to(device)
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
