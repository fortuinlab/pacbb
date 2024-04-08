import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class FClassicObjective(AbstractObjective):
    def __init__(self, kl_penalty: float, delta: float):
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        kl = kl * self._kl_penalty
        kl_ratio = torch.div(kl + np.log((2 * np.sqrt(num_samples)) / self._delta), 2 * num_samples)
        return loss + torch.sqrt(kl_ratio)
