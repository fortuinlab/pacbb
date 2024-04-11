import numpy as np
import torch
from torch import Tensor

from core.objective import AbstractObjective


class MauerObjective(AbstractObjective):
    def __init__(self, kl_penalty: float, delta: float):
        self._kl_penalty = kl_penalty
        self._delta = delta  # confidence value for the training objective

    def calculate(self, loss: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        kl = kl * self._kl_penalty
        kl_ratio = torch.div(kl + 5 / 2 * np.log(num_samples) - np.log(self._delta) + 8, 2 * num_samples - 1)
        return loss + torch.sqrt(kl_ratio)
