from typing import Dict, Callable, List

import torch
import numpy as np
from torch import nn, Tensor

from core.distribution import AbstractVariable


def probabilistic_call(model: nn.Module,
                       data: Tensor,
                       weight_dist: Dict[int, Dict[str, AbstractVariable]],
                       get_layers_func: Callable[[nn.Module], List[nn.Module]],
                       mean: bool = False) -> Tensor:
    for i, layer in enumerate(get_layers_func(model)):
        if not mean:
            layer.weight.data = weight_dist[i]['weight'].sample()
            layer.bias.data = weight_dist[i]['bias'].sample()
    return model(data)


def bounded_probabilistic_call(model: nn.Module,
                               pmin: float,
                               data: Tensor,
                               weight_dist: Dict[int, Dict[str, AbstractVariable]],
                               get_layers_func: Callable[[nn.Module], List[nn.Module]],
                               mean: bool = False) -> Tensor:
    return torch.clamp(probabilistic_call(model, data, weight_dist, get_layers_func, mean), min=np.log(pmin))
