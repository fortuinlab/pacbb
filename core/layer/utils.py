from typing import List, Callable, Iterator, Tuple

import torch.nn as nn

from core.layer import supported_layers


LayerNameT = Tuple[str, ...]


def get_layers(model: nn.Module,
               is_layer_func: Callable[[nn.Module], bool],
               names: List[str]) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    for name, layer in model.named_children():
        if layer is not None:
            yield from get_layers(layer, is_layer_func, names + [name])
        if is_layer_func(layer):
            yield tuple(names + [name]), layer


def is_torch_layer(layer: nn.Module) -> bool:
    return any([isinstance(layer, torch_layer) for torch_layer in supported_layers.LAYER_MAPPING])


def get_torch_layers(model: nn.Module) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    return get_layers(model, is_torch_layer, names=[])


def is_bayesian_torch_layer(layer: nn.Module) -> bool:
    return (hasattr(layer, 'mu_weight') and hasattr(layer, 'rho_weight') and
            hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias'))


def get_bayesian_torch_layers(model: nn.Module) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    return get_layers(model, is_bayesian_torch_layer, names=[])
