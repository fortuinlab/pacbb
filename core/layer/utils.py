from typing import List, Callable

import torch.nn as nn


def get_layers(model: nn.Module, is_layer_func: Callable[[nn.Module], bool]) -> List[nn.Module]:
    # layers = []
    for layer in model.children():
        if layer is not None:
            yield from get_layers(layer, is_layer_func)
        if is_layer_func(layer):
            yield layer
    # return layers


def is_torch_layer(layer: nn.Module) -> bool:
    # return hasattr(layer, 'weight') and hasattr(layer, 'bias')
    return isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)


def get_torch_layers(model: nn.Module) -> List[nn.Module]:
    return get_layers(model, is_torch_layer)


def is_bayesian_torch_layer(layer: nn.Module) -> bool:
    return (hasattr(layer, 'mu_weight') and hasattr(layer, 'rho_weight') and
            hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias'))


def get_bayesian_torch_layers(model: nn.Module) -> List[nn.Module]:
    return get_layers(model, is_bayesian_torch_layer)
