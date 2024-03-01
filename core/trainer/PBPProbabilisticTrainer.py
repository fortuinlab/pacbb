import json
from typing import Dict
from tqdm import trange, tqdm
import torch
from torch.utils.data import dataloader


from core.model.probabilistic import AbstractPBPModel
from core.trainer import AbstractTrainer
from core.trainer.objective import AbstractObjective
from core.trainer.callback import StochasticNLLCallback
from core.utils import logger


class PBPProbabilisticTrainer(AbstractTrainer):
    def __init__(self, device: torch.device):
        super().__init__(device)
        # TODO: add multiple callbacks; add parameter to config
        self._callback = StochasticNLLCallback(device)

    def train(self, model: AbstractPBPModel, optimizer: torch.optim.Optimizer, objective: AbstractObjective, training_config: Dict) -> AbstractPBPModel:
        epochs = training_config['epochs']
        disable_tqdm = training_config['disable_tqdm']
        train_loader = training_config['train_loader']
        val_loader = training_config['val_loader']
        num_samples = training_config['num_samples']

        self._callback.reset()

        for epoch in trange(epochs, disable=disable_tqdm):
            self._step(model,
                       optimizer,
                       objective,
                       epoch,
                       train_loader,
                       num_samples,
                       disable_tqdm)

            self._callback.process(model, val_loader, objective)

        model = self._callback.finish(model, val_loader, objective)

        return model

    def _step(self, model: AbstractPBPModel,
              optimizer: torch.optim.Optimizer,
              objective: AbstractObjective,
              epoch: int,
              train_loader: dataloader.DataLoader,
              num_samples: int,
              disable_tqdm: bool):
        model.train()

        cum_bound, cum_kl, cum_loss_ce, cum_loss_01 = 0.0, 0.0, 0.0, 0.0
        batch, bound, kl, loss_ce, loss_01 = None, None, None, None, None

        for batch, (data, targets) in enumerate(tqdm(train_loader, disable=disable_tqdm)):
            data = data.to(self._device)
            targets = targets.to(self._device)
            model.zero_grad()

            bound, kl, _, loss_ce, loss_01 = objective.train_objective(model, data, targets.long(), num_samples)

            bound.backward()
            optimizer.step()

            cum_bound += bound.item()
            cum_kl += kl.item()
            cum_loss_ce += loss_ce.item()
            cum_loss_01 += loss_01.item()

        logger.debug(self._format_message(epoch, batch+1,
                                          bound.item(), kl.item(), loss_ce.item(), loss_01.item(),
                                          cum_bound, cum_kl, cum_loss_ce, cum_loss_01, round_=3))

    def _format_message(self, epoch: int, batch: int,
                        bound: float, kl: float, loss_ce: float, loss_01: float,
                        cum_bound: float, cum_kl: float, cum_loss_ce: float, cum_loss_01: float, round_: int) -> str:
        message = {
            'epoch': round(epoch, round_),
            'avg_bound': round(cum_bound / batch, round_),
            'avg_kl': round(cum_kl / batch, round_),
            'avg_loss_ce': round(cum_loss_ce / batch, round_),
            'avg_loss_01': round(cum_loss_01 / batch, round_),
            'bound': round(bound, round_),
            'kl': round(kl, round_),
            'loss_ce': round(loss_ce, round_),
            'loss_01': round(loss_01, round_),
        }
        return json.dumps(message)
