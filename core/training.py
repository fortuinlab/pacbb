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
            objective_value.backward()
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Objective: {objective_value}, Loss: {loss}, KL/n: {kl/parameters['num_samples']}")
        if wandb_params is not None and wandb_params["log_wandb"]:
            wandb.log({wandb_params["name_wandb"] + '/Epoch': epoch,
                       wandb_params["name_wandb"] + '/Objective': objective_value,
                       wandb_params["name_wandb"] + '/Loss': loss,
                       wandb_params["name_wandb"] + '/KL-n': kl/parameters['num_samples']})


def train_ivon(model: nn.Module,
               train_loader: torch.utils.data.dataloader.DataLoader,
               val_loader: torch.utils.data.dataloader.DataLoader,
               parameters: Dict[str, Any],
               device: torch.device,
               wandb_params: Dict = None,
        ) -> ivon.IVON:
    if 'sigma' in parameters:
        s0 = 1 / parameters['sigma']
        weight_decay = s0 / parameters['num_samples']
    else:
        weight_decay = 1e-6

    criterion = torch.nn.NLLLoss()
    optimizer = ivon.IVON(model.parameters(), lr=parameters['lr'], ess=parameters['num_samples'],
                          weight_decay=weight_decay)

    if 'seed' in parameters:
        torch.manual_seed(parameters['seed'])
    for epoch in range(parameters['epochs']):
        for i, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            for _ in range(parameters['train_samples']):
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Loss: {loss}")
    return optimizer
