from typing import Callable, Iterator, Tuple

import torch
import numpy as np
from torch import nn, Tensor

from core.distribution.utils import DistributionT
from core.layer import LAYER_MAPPING, AbstractProbLayer
from core.layer.utils import get_torch_layers, LayerNameT


def bounded_call(model: nn.Module,
                 data: Tensor,
                 pmin: float) -> Tensor:
    return torch.clamp(model(data), min=np.log(pmin))


def dnn_to_probnn(model: nn.Module,
                  weight_dist: DistributionT,
                  prior_weight_dist: DistributionT,
                  get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers):
    for name, layer in get_layers_func(model):
        layer_type = type(layer)
        if layer_type in LAYER_MAPPING:
            layer.register_module('_prior_weight_dist', prior_weight_dist[name]['weight'])
            layer.register_module('_prior_bias_dist', prior_weight_dist[name]['bias'])
            layer.register_module('_weight_dist', weight_dist[name]['weight'])
            layer.register_module('_bias_dist', weight_dist[name]['bias'])
            layer.__setattr__('probabilistic_mode', True)
            layer.forward = LAYER_MAPPING[layer_type].forward.__get__(layer, nn.Module)
            layer.probabilistic = AbstractProbLayer.probabilistic.__get__(layer, nn.Module)
            layer.__class__ = LAYER_MAPPING[layer_type]
    model.probabilistic = AbstractProbLayer.probabilistic.__get__(model, nn.Module)
