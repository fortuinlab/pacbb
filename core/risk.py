from typing import Dict, Union
from torch import Tensor

from core.distribution import AbstractVariable
from core.bound import AbstractBound
from core.distribution.utils import compute_kl


def evaluate(bound: AbstractBound,
             loss: Union[Tensor | float],
             posterior: Dict[int, Dict[str, AbstractVariable]],
             prior: Dict[int, Dict[str, AbstractVariable]],
             mc_samples: int,
             bound_samples) -> Dict[str, float]:
    kl = compute_kl(dist1=posterior, dist2=prior)
    risk, empirical_risk = bound.calculate(loss,
                                           mc_samples,
                                           kl,
                                           bound_samples)
    return {'risk': risk.item(), 'empirical_risk': empirical_risk.item(), 'kl': kl/bound_samples}
