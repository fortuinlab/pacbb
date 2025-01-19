"""
## Overview
The `core` package provides foundational components for the PAC-Bayes framework.

## Subpackages and Modules
- **distribution/** for modeling probability distributions over NN parameters
- **layer/** for probabilistic neural network layers
- **bound/** for PAC-Bayes bounds
- **objective/** for combining empirical loss and KL terms
- **split_strategy/** for partitioning datasets (prior/posterior/bound)
- **utils/** for miscellaneous helpers
- Higher-level modules (*loss.py*, *metric.py*, *model.py*, *risk.py*, *training.py*)
  that stitch these components together.

## Usage
By assembling the pieces within `core`, you can build, train, and evaluate
probabilistic neural networks under the PAC-Bayes paradigm.
"""
