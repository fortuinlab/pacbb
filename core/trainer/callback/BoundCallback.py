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


class BoundCallback(AbstractCallback):
    def __init__(self, freq_test, device: torch.device):
        super().__init__(device)
        self._freq_test = freq_test
        self._best_val_risk = np.inf
        self._best_model = None

    def reset(self) -> None:
        self._best_val_risk = np.inf
        self._best_model = None

    def process(
        self,
        epoch: int,
        model: AbstractPBPModel,
        loader: data.Dataset,
        objective: AbstractObjective,
    ) -> None:
        if loader is not None and (epoch+1) % self._freq_test == 0:
            val_risk, _, _ = ModelEvaluator.evaluate_stochastic(
                model, loader, objective, 1, self._device
            )
            if val_risk < self._best_val_risk:
                logger.info(
                    json.dumps(
                        {
                            "nll_loss": round(val_risk, 3),
                            "best_nll_loss": round(self._best_val_risk, 3),
                        }
                    )
                )
                self._best_model = copy.deepcopy(model)
                self._best_val_risk = val_risk

    def finish(
        self,
        model: AbstractPBPModel,
        loader: data.Dataset,
        objective: AbstractObjective,
    ) -> AbstractPBPModel:
        if loader is not None and self._best_model is not None:
            return self._best_model
        else:
            return model
