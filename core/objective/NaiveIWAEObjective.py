import torch
from torch import Tensor
from typing import List

from core.objective import AbstractSamplingObjective


class NaiveIWAEObjective(AbstractSamplingObjective):
    def calculate(self, losses: List[Tensor], kl: Tensor, num_samples: float) -> Tensor:
        assert self.n == len(losses)
        loss = sum(losses) / self.n
        return loss + self._kl_penalty * (kl / num_samples)
