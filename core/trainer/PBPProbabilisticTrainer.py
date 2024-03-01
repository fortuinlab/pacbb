from typing import Dict
from tqdm import trange, tqdm
import torch
from torch.utils.data import dataloader

from core.model.probabilistic import AbstractPBPModel
from core.trainer import AbstractTrainer
from core.trainer.objective import AbstractObjective


class PBPProbabilisticTrainer(AbstractTrainer):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def train(self, model: AbstractPBPModel, optimizer: torch.optim.Optimizer, objective: AbstractObjective, training_config: Dict) -> AbstractPBPModel:
        epochs = training_config['epochs']
        disable_tqdm = training_config['disable_tqdm']
        train_loader = training_config['train_loader']

        for epoch in trange(epochs, disable=disable_tqdm):
            self._step(model,
                       optimizer,
                       objective,
                       train_loader)

    def _step(self, model: AbstractPBPModel,
              optimizer: torch.optim.Optimizer,
              objective: AbstractObjective,
              train_loader: dataloader.DataLoader):
        model.train()

        for batch_id, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(self._device)
            targets = targets.to(self._device)

            model.zero_grad()

            bound, kl, _, loss_ce, loss_01 = objective.train_objective(model, data, targets.long())