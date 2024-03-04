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
        self._risk_name = 'risk_01'  # todo: move to parameters?
        self._freq_test = freq_test
        self._best_risk = np.inf
        self._best_model = None

    def reset(self) -> None:
        self._best_risk = np.inf
        self._best_model = None

    def process(
        self,
        epoch: int,
        model: AbstractPBPModel,
        train_loader: data.Dataset,
        val_loader: data.Dataset,
        objective: AbstractObjective,
    ) -> None:
        if val_loader is not None and (epoch+1) % self._freq_test == 0:
            evaluation_result_dict = ModelEvaluator.evaluate_risk(
                model=model,
                loader=train_loader,
                bound_loader=val_loader,
                objective=objective,
                device=self._device,
            )
            logger.info(
                json.dumps(
                    {
                        f"{self._risk_name}_risk": round(evaluation_result_dict[self._risk_name], 5),
                        f"{self._risk_name}_best_risk": round(self._best_risk, 5),
                    }
                )
            )
            if evaluation_result_dict[self._risk_name] < self._best_risk:
                logger.info('Updated best model')
                self._best_model = copy.deepcopy(model)
                self._best_risk = evaluation_result_dict[self._risk_name]

    def finish(
        self,
        model: AbstractPBPModel,
        train_loader: data.Dataset,
        val_loader: data.Dataset,
        objective: AbstractObjective,
    ) -> AbstractPBPModel:
        if val_loader is not None and self._best_model is not None:
            return self._best_model
        else:
            return model
