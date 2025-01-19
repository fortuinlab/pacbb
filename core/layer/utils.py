from typing import List, Callable, Iterator, Tuple

import torch.nn as nn

from core.layer import supported_layers


LayerNameT = Tuple[str, ...]


def get_layers(model: nn.Module,
               is_layer_func: Callable[[nn.Module], bool],
               names: List[str]) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    """
    Recursively traverse a PyTorch model to find layers matching a given criterion.

    This function performs a depth-first search over children of `model`.
    If `is_layer_func(layer)` is True, yield the path of layer names (as a tuple)
    and the layer object.

    Args:
        model (nn.Module): The PyTorch model or submodule to traverse.
        is_layer_func (Callable): A predicate function that returns True if a layer
            matches the criterion (e.g., belongs to a certain set of layer types).
        names (List[str]): Accumulates the hierarchical names as we recurse.

    Yields:
        Iterator[Tuple[LayerNameT, nn.Module]]: Tuples of (layer_name_path, layer_object).
    """
    for name, layer in model.named_children():
        if layer is not None:
            yield from get_layers(layer, is_layer_func, names + [name])
        if is_layer_func(layer):
            yield tuple(names + [name]), layer


def is_torch_layer(layer: nn.Module) -> bool:
    """
    Check if the given layer is a supported PyTorch layer in the framework.

    Args:
        layer (nn.Module): A PyTorch layer or module.

    Returns:
        bool: True if the layer's type is one of the framework's supported mappings.
    """
    return any([isinstance(layer, torch_layer) for torch_layer in supported_layers.LAYER_MAPPING])


def get_torch_layers(model: nn.Module) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    """
    Yield all supported PyTorch layers in the model.

    Args:
        model (nn.Module): The PyTorch model to traverse.

    Returns:
        Iterator[Tuple[LayerNameT, nn.Module]]: Each tuple is (path_of_names, layer).
    """
    return get_layers(model, is_torch_layer, names=[])


def is_bayesian_torch_layer(layer: nn.Module) -> bool:
    """
    Check if the layer belongs to a BayesianTorch-style layer,
    identified by having 'mu_weight', 'rho_weight', 'mu_bias', 'rho_bias'
    or the 'kernel' equivalent attributes.

    Args:
        layer (nn.Module): A PyTorch module.

    Returns:
        bool: True if the layer has BayesianTorch parameter attributes.
    """
    return ((hasattr(layer, 'mu_weight') and hasattr(layer, 'rho_weight') and
             hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias')) or
            (hasattr(layer, 'mu_kernel') and hasattr(layer, 'rho_kernel') and
             hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias')))


def get_bayesian_torch_layers(model: nn.Module) -> Iterator[Tuple[LayerNameT, nn.Module]]:
    """
    Yield all layers in the model recognized as BayesianTorch layers,
    i.e., those containing 'mu_weight', 'rho_weight', etc.

    Args:
        model (nn.Module): The PyTorch model to traverse.

    Returns:
        Iterator[Tuple[LayerNameT, nn.Module]]: (layer_name_path, layer_object) for each Bayesian layer.
    """
    return get_layers(model, is_bayesian_torch_layer, names=[])
