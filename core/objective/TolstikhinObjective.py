import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class TolstikhinObjective(AbstractObjective):
    def __init__(self, kl_penalty: float, delta: float):
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        kl = kl * self._kl_penalty
        second_term = 2 * loss * torch.div(kl + np.log(2 * num_samples) - np.log(self._delta),
                                           2 * num_samples)
        third_term = 2 * torch.div(kl + np.log(2 * num_samples) - np.log(self._delta),
                                   2 * num_samples)
        return loss + torch.sqrt(second_term) + third_term
