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
            layer.__class__ = LAYER_MAPPING[layer_type]
    model.probabilistic = AbstractProbLayer.probabilistic.__get__(model, nn.Module)


def update_dist(model: nn.Module,
                weight_dist: DistributionT = None,
                prior_weight_dist: DistributionT = None,
                get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers):
    if weight_dist is not None:
        for (name, distribution), (_, layer) in zip(weight_dist.items(), get_layers_func(model)):
            layer_type = type(layer)
            if layer_type in LAYER_MAPPING.values():
                layer.__setattr__('_weight_dist', distribution['weight'])
                layer.__setattr__('_bias_dist', distribution['bias'])

    if prior_weight_dist is not None:
        for (name, distribution), (_, layer) in zip(prior_weight_dist.items(), get_layers_func(model)):
            layer_type = type(layer)
            if layer_type in LAYER_MAPPING.values():
                layer.__setattr__('_prior_weight_dist', distribution['weight'])
                layer.__setattr__('_prior_bias_dist', distribution['bias'])
