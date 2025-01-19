"""
The `layer` subpackage contains probabilistic layer definitions and related utilities.

Highlights:
  - `AbstractProbLayer` base class, enabling weight/bias sampling via posterior distributions.
  - Probabilistic versions of PyTorch layers (e.g., `ProbConv2d`, `ProbLinear`).
  - Helper methods for layer introspection (`utils.py`).

These layers replace deterministic operations with sampling, enabling Bayesian
inference under the PAC-Bayes framework.
"""


from core.layer.AbstractProbLayer import AbstractProbLayer
from core.layer.ProbLinear import ProbLinear
from core.layer.ProbConv2d import ProbConv2d
from core.layer.ProbBatchNorm1d import ProbBatchNorm1d
from core.layer.ProbBatchNorm2d import ProbBatchNorm2d
from core.layer.supported_layers import LAYER_MAPPING
