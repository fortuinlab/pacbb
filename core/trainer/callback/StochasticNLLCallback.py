import torch
import numpy as np
from torch.utils import data
import json
import copy

from core.trainer.callback import AbstractCallback
from core.model.probabilistic import AbstractPBPModel
from core.trainer.objective import AbstractObjective
from core.utils import logger
from core.model.evaluation import ModelEvaluator


class StochasticNLLCallback(AbstractCallback):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self._best_val_nll_loss = np.inf
        self._best_model = None

    def reset(self) -> None:
        self._best_val_nll_loss = np.inf
        self._best_model = None

    def process(self, model: AbstractPBPModel,
                loader: data.Dataset,
                objective: AbstractObjective) -> None:
        if loader is not None:
            val_loss, _, _ = ModelEvaluator.evaluate_stochastic(model, loader, objective, 1, self._device)
            if val_loss < self._best_val_nll_loss:
                logger.info(json.dumps({'nll_loss': round(val_loss, 3),
                                        'best_nll_loss': round(self._best_val_nll_loss, 3)}))
                self._best_model = copy.deepcopy(model)
                self._best_val_nll_loss = val_loss

    def finish(self, model: AbstractPBPModel, loader: data.Dataset, objective: AbstractObjective) -> AbstractPBPModel:
        if loader is not None and self._best_model is not None:
            return self._best_model
        else:
            return model
