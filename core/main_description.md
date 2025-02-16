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

In practice, you might have a factory of losses:
```python
#Example: Instantiating losses and LossFactory
loss_factory = LossFactory()
losses = {
    "nll_loss": loss_factory.create("nll_loss"),
    "scaled_nll_loss": loss_factory.create("scaled_nll_loss"),
    "01_loss": loss_factory.create("01_loss")
}
```
Different variants can be plugged in, as long as they are implemented under the `AbstractLoss` interface (which ensures a forward pass that outputs a bounded value).

### Metrics

Metrics serve solely as evaluation tools, such as classification accuracy, the F1-score, or custom-defined measures. While they do not necessarily have bounded values or appear explicitly in the PAC-Bayes inequality, they provide essential insights for assessing model performance.

```python
#Example: Creating metrics for evaluation (not necessarily bounded).
metric_factory = MetricFactory()
metrics = {
    "accuracy_micro_metric": metric_factory.create("accuracy_micro_metric"),
    "accuracy_macro_metric": metric_factory.create("accuracy_macro_metric"),
    "f1_micro_metric": metric_factory.create("f1_micro_metric"),
    "f1_macro_metric": metric_factory.create("f1_macro_metric")
}
```

These can be accuracy, precision, confusion matrices, etc.

### Bounds and the Role of KL Divergence

PAC-Bayes bounds typically look like:

$$
\mathbb{E}_{\theta\sim\rho}[R(\theta)]
\,\le\,
\mathbb{E}_{\theta\sim\rho}[r(\theta)]
\;+\;
\mathrm{Complexity}(KL(\rho \| \pi),\, n, \delta),
$$

where:

- $KL(\rho \|\pi)$ measures how far the posterior $\rho$ is from the prior $\pi$.
- $n$ is the effective number of data samples used in the bound. In a data-splitting scenario, it often corresponds to the size of the bound set, and in code it is usually deduced from the `bound_loader` (for example by `len(bound_loader.dataset)`).
- $\delta$ is a confidence parameter. The bound holds with probability at least $1 - \delta$.

In the snippet below, `bound_delta` corresponds to the theoretical $\delta$. The parameter `loss_delta` is a separate hyperparameter for bounding or adjusting the loss term in practice. For instance, one might cap a negative log-likelihood at $\log(1/p_{\min})$ or include a data-driven threshold. While usage can vary by codebase, typically:

- `bound_delta` is the $\delta$ from the PAC-Bayes statement.
- `loss_delta` handles or scales losses, especially when the loss must be numerically bounded.

```python
#Example: Instantiating a PAC-Bayes Bound with delta parameters
bound_factory = BoundFactory()
bounds = {
    "kl": bound_factory.create(
        "kl",
        bound_delta=0.025,   
        loss_delta=0.01
    ),
    "mcallister": bound_factory.create(
        "mcallister",
        bound_delta=0.025,
        loss_delta=0.01
    )
}
```

### Data Splitting & Prior Selection

To build an informed prior without violating the PAC-Bayes assumptions, we split the available data into three subsets: $\mathcal{S}_{\text{prior}}$, $\mathcal{S}_{\text{bound}}$, and $\mathcal{S}_{\text{posterior}}$. Specifically:

- The prior is trained solely on $\mathcal{S}_{\text{prior}}$. 
- The PAC-Bayes bound is computed later on $\mathcal{S}_{\text{bound}}$ only, ensuring that the prior was chosen independently of these samples.
- The posterior can be refined on additional data (potentially uniting $\mathcal{S}_{\text{prior}}$ and $\mathcal{S}_{\text{posterior}}$) after the prior has been established. 

By specifying `prior_type="learnt"`, we indicate in the code below that we intend to select or train a prior distribution from the $\mathcal{S}_{\text{prior}}$ samples, rather than fixing a data-independent prior. The fraction `prior_percent=0.7` means that 70% of the `train_percent` samples go to the prior, while the remaining 30% form $\mathcal{S}_{\text{bound}}$ (with additional splitting if validation/test sets are used). Once split, $\mathcal{S}_{\text{prior}}$ is passed to a training to learn the prior distribution’s parameters.

```python
#Example: Splitting dataset into prior/posterior/bound (plus val/test).
data_loader_factory = DataLoaderFactory()
loader = data_loader_factory.create(
    "cifar10",
    dataset_path="./data/cifar10"
)

strategy = PBPSplitStrategy(
    prior_type="learnt",   # we want to learn a prior from data
    train_percent=1.0,     # use entire dataset for (prior + bound)
    val_percent=0.0,       # no separate validation here
    prior_percent=0.7,     # 70% of training data for prior, 30% for bound
    self_certified=True    # ensures the prior set is disjoint from bound set
)
strategy.split(loader, split_config={
    "batch_size": 250, 
    "dataset_loader_seed": 112,
    "seed": 111
})
```


---

## Links

- **Documentation**: [https://yauhenii.github.io/pacbb/core.html](https://yauhenii.github.io/pacbb/core.html)
- **PyPI**: [https://pypi.org/project/pacbb/](https://pypi.org/project/pacbb/)
- **Source Code**: [https://github.com/yauhenii/pacbb](https://github.com/yauhenii/pacbb)
