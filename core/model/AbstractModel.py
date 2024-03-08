from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class AbstractModel(nn.Module, ABC):
    @staticmethod
    def output_transform(x: Tensor, clamping: bool, p_min: float) -> Tensor:
        output = F.log_softmax(x, dim=1)
        if clamping:
            output = torch.clamp(output, np.log(p_min))
        return output
