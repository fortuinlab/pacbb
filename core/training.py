from typing import Dict, Any
import logging
import wandb

import ivon
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from core.distribution.utils import compute_kl, DistributionT
from core.objective import AbstractObjective, AbstractSamplingObjective, IWAEObjective
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
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            lr=parameters['lr'],
    #                            momentum=parameters['momentum'])

    #just to try
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=parameters['lr'])
    if 'seed' in parameters:
        torch.manual_seed(parameters['seed'])

    loss = None
    kl = None
    objective_value = None

    dataset_size = len(train_loader.dataset) # new added

    for epoch in range(parameters['epochs']):
        for i, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if isinstance(objective, AbstractObjective):
                if 'pmin' in parameters:
                    output = bounded_call(model, data, parameters['pmin'])
                else:
                    output = model(data)
                loss = criterion(output, target)
                kl = compute_kl(posterior, prior)
                objective_value = objective.calculate(loss, kl, parameters['num_samples'])
            elif isinstance(objective, AbstractSamplingObjective):
                losses = []
                for j in range(objective.n):
                    if 'pmin' in parameters:
                        output = bounded_call(model, data, parameters['pmin'])
                    else:
                        output = model(data)
                    losses.append(criterion(output, target))
                kl = compute_kl(posterior, prior)
                objective_value = objective.calculate(losses, kl, parameters['num_samples'])
                loss = sum(losses) / objective.n
            elif isinstance(objective, IWAEObjective):
                objective_value = objective.calculate(model,
                                                      data,
                                                      target,
                                                      epoch=epoch,
                                                      batch_idx=i,
                                                      dataset_size= dataset_size,
                                                      pmin=parameters.get('pmin', None),
                                                      wandb_params=wandb_params)
                with torch.no_grad():
                    loss = criterion(model(data), target)
                    kl = compute_kl(posterior, prior)
            else:
                raise ValueError(f'Invalid objective type: {type(objective)}')
            objective_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        logging.info(f"Epoch: {epoch}, Objective: {objective_value}, Loss: {loss}, KL/n: {kl/parameters['num_samples']}")
        if wandb_params is not None and wandb_params["log_wandb"]:
            wandb.log({wandb_params["name_wandb"] + '/Epoch': epoch,
                       wandb_params["name_wandb"] + '/Objective': objective_value,
                       wandb_params["name_wandb"] + '/Loss': loss,
                       wandb_params["name_wandb"] + '/KL-n': kl/parameters['num_samples']})
