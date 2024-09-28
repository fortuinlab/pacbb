import math
from typing import List, Union, Type, Callable, Dict, Tuple, Iterator

import ivon
import torch
from torch import nn, Tensor

from core.layer.utils import get_torch_layers, LayerNameT, get_bayesian_torch_layers
from core.distribution import AbstractVariable


DistributionT = Dict[LayerNameT, Dict[str, AbstractVariable]]


def from_ivon(model: nn.Module,
              optimizer: ivon.IVON,
              distribution: Type[AbstractVariable],
              requires_grad: bool = True,
              get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers,
              ) -> DistributionT:
    distributions = {}
    i = 0
    shift = 0
    weights = optimizer.param_groups[0]['params']
    hessians = optimizer.param_groups[0]['hess']
    weight_decay = optimizer.param_groups[0]["weight_decay"]
    ess = optimizer.param_groups[0]["ess"]
    sigma = 1 / (ess * (hessians + weight_decay)).sqrt()
    rho = torch.log(torch.exp(sigma) - 1)

    for name, layer in get_layers_func(model):
        if layer.weight is not None:
            weight_cutoff = shift + math.prod(layer.weight.shape)
            weight_distribution = distribution(mu=weights[i],
                                               rho=rho[shift: weight_cutoff].reshape(*layer.weight.shape),
                                               mu_requires_grad=requires_grad,
                                               rho_requires_grad=requires_grad)
            shift = weight_cutoff
            i+=1
        else:
            weight_distribution = None
        if layer.bias is not None:
            bias_cutoff = shift + math.prod(layer.bias.shape)
            bias_distribution = distribution(mu=weights[i],
                                             rho=rho[shift: bias_cutoff].reshape(*layer.bias.shape),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
            shift = bias_cutoff
            i+=1
        else:
            bias_distribution = None
        distributions[name] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def from_flat_rho(model: nn.Module,
                  rho: Union[Tensor, List[float]],
                  distribution: Type[AbstractVariable],
                  requires_grad: bool = True,
                  get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers,
                  ) -> DistributionT:
    distributions = {}
    shift = 0
    for name, layer in get_layers_func(model):
        if layer.weight is not None:
            # weight_cutoff = shift + layer.out_features * layer.in_features
            weight_cutoff = shift + math.prod(layer.weight.shape)
            weight_distribution = distribution(mu=layer.weight,
                                               rho=rho[shift: weight_cutoff].reshape(*layer.weight.shape),
                                               mu_requires_grad=requires_grad,
                                               rho_requires_grad=requires_grad)
        else:
            weight_distribution = None
        if layer.bias is not None:
            # bias_cutoff = weight_cutoff + layer.out_features
            bias_cutoff = weight_cutoff + math.prod(layer.bias.shape)
            bias_distribution = distribution(mu=layer.bias,
                                             rho=rho[weight_cutoff: bias_cutoff].reshape(*layer.bias.shape),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
            shift = bias_cutoff
        else:
            bias_distribution = None
            shift = weight_cutoff
        distributions[name] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def _from_any(model: nn.Module,
              distribution: Type[AbstractVariable],
              requires_grad: bool,
              get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]],
              weight_mu_fill_func: Callable[[nn.Module], Tensor],
              weight_rho_fill_func: Callable[[nn.Module], Tensor],
              bias_mu_fill_func: Callable[[nn.Module], Tensor],
              bias_rho_fill_func: Callable[[nn.Module], Tensor],
              weight_exists: Callable[[nn.Module], bool] = lambda layer: hasattr(layer, 'weight'),
              bias_exists: Callable[[nn.Module], bool] = lambda layer: hasattr(layer, 'bias')
              ) -> DistributionT:
    distributions = {}
    for name, layer in get_layers_func(model):
        if weight_exists(layer) and layer.weight is not None:
            weight_distribution = distribution(mu=weight_mu_fill_func(layer),
                                               rho=weight_rho_fill_func(layer),
                                               mu_requires_grad=requires_grad,
                                               rho_requires_grad=requires_grad)
        else:
            weight_distribution = None
        if bias_exists(layer) and layer.bias is not None:
            bias_distribution = distribution(mu=bias_mu_fill_func(layer),
                                             rho=bias_rho_fill_func(layer),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
        else:
            bias_distribution = None
        distributions[name] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def from_random(model: nn.Module,
                rho: Tensor,
                distribution: Type[AbstractVariable],
                requires_grad: bool = True,
                get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers,
                ) -> DistributionT:
    def get_truncated_normal_fill_tensor(layer: nn.Module) -> Tensor:
        t = torch.Tensor(*layer.weight.shape)
        if hasattr(layer, 'weight') and layer.weight is not None:
            in_features = math.prod(layer.weight.shape[1:])
        else:
            raise ValueError(f'Unsupported layer of type: {type(layer)}')
        w = 1 / math.sqrt(in_features)
        return truncated_normal_fill_tensor(t, 0, w, -2 * w, 2 * w)
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_mu_fill_func=get_truncated_normal_fill_tensor,
                     weight_rho_fill_func=lambda layer: torch.ones(*layer.weight.shape) * rho,
                     bias_mu_fill_func=lambda layer: torch.zeros(*layer.bias.shape),
                     bias_rho_fill_func=lambda layer: torch.ones(*layer.bias.shape) * rho)


def from_zeros(model: nn.Module,
               rho: Tensor,
               distribution: Type[AbstractVariable],
               requires_grad: bool = True,
               get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers,
               ) -> DistributionT:
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_mu_fill_func=lambda layer: torch.zeros(*layer.weight.shape),
                     weight_rho_fill_func=lambda layer: torch.ones(*layer.weight.shape) * rho,
                     bias_mu_fill_func=lambda layer: torch.zeros(*layer.bias.shape),
                     bias_rho_fill_func=lambda layer: torch.ones(*layer.bias.shape) * rho)


def from_layered(model: torch.nn.Module,
                 attribute_mapping: dict[str, str],
                 distribution: Type[AbstractVariable],
                 requires_grad: bool = True,
                 get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_torch_layers,
                 ) -> DistributionT:
    return _from_any(model, distribution, requires_grad, get_layers_func,
                     weight_exists=lambda layer: hasattr(layer, attribute_mapping['weight_mu']) and hasattr(layer, attribute_mapping['weight_rho']),
                     bias_exists=lambda layer: hasattr(layer, attribute_mapping['weight_mu']) and hasattr(layer, attribute_mapping['weight_rho']),
                     weight_mu_fill_func=lambda layer: layer.__getattr__(attribute_mapping['weight_mu']).detach().clone(),
                     weight_rho_fill_func=lambda layer: layer.__getattr__(attribute_mapping['weight_rho']).detach().clone(),
                     bias_mu_fill_func=lambda layer: layer.__getattr__(attribute_mapping['bias_mu']).detach().clone(),
                     bias_rho_fill_func=lambda layer: layer.__getattr__(attribute_mapping['bias_rho']).detach().clone())


def from_bnn(model: nn.Module,
             distribution: Type[AbstractVariable],
             requires_grad: bool = True,
             get_layers_func: Callable[[nn.Module], Iterator[Tuple[LayerNameT, nn.Module]]] = get_bayesian_torch_layers,
             ) -> DistributionT:
    distributions = {}
    for name, layer in get_layers_func(model):
        if hasattr(layer, 'mu_weight') and hasattr(layer, 'rho_weight'):
            weight_distribution = distribution(mu=layer.__getattr__('mu_weight').detach().clone(),
                                               rho=layer.__getattr__('rho_weight').detach().clone(),
                                               mu_requires_grad=requires_grad,
                                               rho_requires_grad=requires_grad)
        elif hasattr(layer, 'mu_kernel') and hasattr(layer, 'rho_kernel'):
            weight_distribution = distribution(mu=layer.__getattr__('mu_kernel').detach().clone(),
                                               rho=layer.__getattr__('rho_kernel').detach().clone(),
                                               mu_requires_grad=requires_grad,
                                               rho_requires_grad=requires_grad)
        else:
            weight_distribution = None
        if hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias'):
            bias_distribution = distribution(mu=layer.__getattr__('mu_bias').detach().clone(),
                                             rho=layer.__getattr__('rho_bias').detach().clone(),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
        else:
            bias_distribution = None
        distributions[name] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def from_copy(dist: DistributionT,
              distribution: Type[AbstractVariable],
              requires_grad: bool = True,
              ) -> DistributionT:
    distributions = {}
    for name, layer in dist.items():
        weight_distribution = distribution(mu=layer['weight'].mu.detach().clone(),
                                           rho=layer['weight'].rho.detach().clone(),
                                           mu_requires_grad=requires_grad,
                                           rho_requires_grad=requires_grad)
        if layer['bias'] is not None:
            bias_distribution = distribution(mu=layer['bias'].mu.detach().clone(),
                                             rho=layer['bias'].rho.detach().clone(),
                                             mu_requires_grad=requires_grad,
                                             rho_requires_grad=requires_grad)
        else:
            bias_distribution = None
        distributions[name] = {'weight': weight_distribution, 'bias': bias_distribution}
    return distributions


def compute_kl(dist1: DistributionT, dist2: DistributionT) -> Tensor:
    kl_list = []
    for idx in dist1:
        for key in dist1[idx]:
            if dist1[idx][key] is not None and dist2[idx][key] is not None:
                kl = dist1[idx][key].compute_kl(dist2[idx][key])
                kl_list.append(kl)
    return torch.stack(kl_list).sum()


def compute_standard_normal_cdf(x: float) -> float:
    """
    Compute the standard normal cumulative distribution function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The cumulative distribution function value at x.
    """
    # TODO: replace with numpy
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def truncated_normal_fill_tensor(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    # TODO: refactor
    with torch.no_grad():
        # Get upper and lower cdf values
        l = compute_standard_normal_cdf((a - mean) / std)
        u = compute_standard_normal_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
