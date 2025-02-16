## Overview
The `core` package provides foundational components for the PAC-Bayes framework. By assembling the pieces within `core`, you can build, train, and evaluate
probabilistic neural networks under the PAC-Bayes paradigm.

## Subpackages and Modules
- **distribution/** for modeling probability distributions over NN parameters
- **layer/** for probabilistic neural network layers
- **bound/** for PAC-Bayes bounds
- **objective/** for combining empirical loss and KL terms
- **split_strategy/** for partitioning datasets (prior/posterior/bound)
- **utils/** for miscellaneous helpers
- Higher-level modules (*loss.py*, *metric.py*, *model.py*, *risk.py*, *training.py*)
  that stitch these components together.

## PAC-Bayes Framework: Theory and Practice

**Note:** The explanations below are demonstrated using an actual working script. You can directly run it from the repository:  
[https://github.com/Yauhenii/pacbb/blob/main/scripts/generic_train.py](https://github.com/Yauhenii/pacbb/blob/main/scripts/generic_train.py)  
This script implements a complete training pipeline including prior/posterior training, data splitting, and final PAC-Bayes bound certification.

---

### Introduction

This framework provides tools to build, train, and evaluate probabilistic neural networks under the PAC-Bayes paradigm, combining theoretical guarantees with practical utility. In PAC-Bayes, we treat each neural network parameter as a random variable, so the network becomes a distribution over possible parameter values. The approach offers generalization bounds of the form:

$$
\mathbb{E}_{\theta \sim \rho} [R(\theta)] 
\;\;\leq\;\;
\underbrace{ \mathbb{E}_{\theta \sim \rho} [r(\theta)] }_{\text{empirical loss}}
\;+\;\mathrm{Complexity}\bigl(KL(\rho \| \pi),\, n\bigr),
$$

where $\rho$ is a posterior distribution over parameters, 
$\pi$ is a prior, $r(\theta)$ is the empirical loss 
(e.g., on a training or bound dataset), and 
$R(\theta)$ is the (unknown) true risk.

In practice, one typically:

1. Chooses a model architecture in PyTorch and attaches probabilistic distributions (for example, Gaussians) to its parameters.  
2. Splits data to allow for prior training, posterior training, and a dedicated set for bound evaluation.  
3. Trains the prior distribution, optionally measuring and bounding its risk.  
4. Initializes and refines the posterior distribution from the learned prior.  
5. Computes or certifies the final PAC-Bayes bound on the bound set.

Below, we break down each code component in an example pipeline. Along the way, we point out key theoretical ideas (for example, the bounded-loss requirement, data splitting for data-dependent priors, or the role of KL divergence in each objective). This corresponds to the working script `generic_train.py` from our repository.


### Losses in PAC-Bayes

A core requirement of PAC-Bayes is that the loss function must be bounded, typically in $[0,1]$. For classification, the 0-1 loss is a natural choice but is not differentiable. Instead, implementations often employ a bounded negative log-likelihood:

$$
l(f(X), Y) = \frac{-\log \max\{P(Y\mid f(X)), p_{\min}\}}{\log(1/p_{\min})}
$$

so that $l$ remains in $[0,1]$.


---

## Links

- **Documentation**: [https://yauhenii.github.io/pacbb/core.html](https://yauhenii.github.io/pacbb/core.html)
- **PyPI**: [https://pypi.org/project/pacbb/](https://pypi.org/project/pacbb/)
- **Source Code**: [https://github.com/yauhenii/pacbb](https://github.com/yauhenii/pacbb)
