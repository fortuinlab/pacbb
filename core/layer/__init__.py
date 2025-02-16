"""
## Overview
Contains probabilistic versions of common neural network layers and the
infrastructure to attach distributions to them.

## Contents
- **AbstractProbLayer**: A base class managing probabilistic_mode and sampling
- **ProbConv2d, ProbLinear, ProbBatchNorm1d, ProbBatchNorm2d**:
  Probabilistic layers derived from PyTorch
- **utils.py** for layer inspection and traversal

These layers replace deterministic parameters with sampled ones,
supporting Bayesian inference under the PAC-Bayes framework.
"""

from core.layer.AbstractProbLayer import AbstractProbLayer
from core.layer.ProbBatchNorm1d import ProbBatchNorm1d
from core.layer.ProbBatchNorm2d import ProbBatchNorm2d
from core.layer.ProbConv2d import ProbConv2d
from core.layer.ProbLinear import ProbLinear
from core.layer.supported_layers import LAYER_MAPPING
