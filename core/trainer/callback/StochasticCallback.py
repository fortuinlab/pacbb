import copy
import json

import numpy as np
import torch
from torch.utils import data

from core.model.evaluation import ModelEvaluator
from core.model.probabilistic import AbstractPBPModel
from core.trainer.callback import AbstractCallback
from core.trainer.objective import AbstractObjective
from core.utils import logger


class StochasticNLLCallback(AbstractCallback):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self._best_loss = np.inf
        self._best_model = None

    def reset(self) -> None:
        self._best_loss = np.inf
        self._best_model = None

    def process(
        self,
        epoch: int,
        model: AbstractPBPModel,
        train_loader: data.Dataset,
        val_loader: data.Dataset,
        objective: AbstractObjective,
    ) -> None:
        if val_loader is not None:
            loss, _, _ = ModelEvaluator.evaluate_stochastic(
                model, val_loader, objective, 1, self._device
            )
            if loss < self._best_loss:
                logger.info(
                    json.dumps(
                        {
                            "loss": round(loss, 5),
                            "best_loss": round(self._best_loss, 5),
                        }
                    )
                )
                self._best_model = copy.deepcopy(model)
                self._best_loss = loss

    def finish(
        self,
        epoch: int,
        model: AbstractPBPModel,
        train_loader: data.Dataset,
        val_loader: data.Dataset,
        objective: AbstractObjective,
    ) -> AbstractPBPModel:
        if val_loader is not None and self._best_model is not None:
            return self._best_model
        else:
            return model
