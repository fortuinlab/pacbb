from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from core.model import AbstractModel
from core.utils import KLDivergenceInterface

class AbstractPBPModel(AbstractModel, KLDivergenceInterface, ABC):
    def __init__(
        self,
        model_weight_distribution: str,
        sigma: float,
        weight_initialization_method: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self._model_weight_distribution = model_weight_distribution
        self._sigma = sigma
        self._weight_initialization_method = weight_initialization_method
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._device = device
        self.kl_div = None

    @staticmethod
    def output_transform(x: torch.Tensor, clamping: bool, p_min: float = 1e-4):
        output = F.log_softmax(x, dim=1)
        if clamping:
            output = torch.clamp(output, np.log(p_min))
        return output

    @abstractmethod
    def compute_kl(self, recompute: bool = True) -> torch.Tensor:
        pass