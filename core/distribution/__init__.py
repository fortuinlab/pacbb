"""
## Overview
Provides classes and utilities for representing probability distributions
over neural network parameters.

## Key Components
- **AbstractVariable**: a base interface for distribution variables
  (e.g., GaussianVariable).
- **Utility functions** for constructing, copying, and computing KL divergences
  between distributions.

These distributions form the core representation of uncertainty in a
PAC-Bayes model.
"""

from core.distribution.AbstractVariable import AbstractVariable
from core.distribution.GaussianVariable import GaussianVariable
from core.distribution.LaplaceVariable import LaplaceVariable
