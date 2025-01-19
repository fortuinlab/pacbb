"""
The `distribution` subpackage provides classes for representing probability
distributions over neural network parameters.

Key components:
  - `AbstractVariable` for defining distributions (e.g., GaussianVariable),
  - Utilities for constructing, copying, and computing KL divergences between distributions.

These distributions form the core representation of uncertainty in a PAC-Bayes model.
"""

from core.distribution.AbstractVariable import AbstractVariable
from core.distribution.GaussianVariable import GaussianVariable
