from typing import Dict, Union, List
from torch import Tensor

from app.model.probabilistic.distribution import AbstractVariable
from core.bound import AbstractBound


class BoundEvaluator:
    @staticmethod
    def evaluate_risk(bound: AbstractBound,
                      loss: Union[Tensor | float],
                      posterior: List[AbstractVariable],
                      prior: List[AbstractVariable],
                      num_mc_samples: int,
                      bound_num_samples) -> Dict[str, float]:
        kl = sum([po.compute_kl(pr).item() for po, pr in zip(posterior, prior)])
        risk, empirical_risk = bound.calculate(loss,
                                               num_mc_samples,
                                               kl,
                                               bound_num_samples)
        return {'risk': risk.item(), 'empirical_risk': empirical_risk.item(), 'kl': kl/bound_num_samples}
