[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://yauhenii.github.io/pacbb/core.html)
[![PyPI](https://img.shields.io/pypi/v/pacbb.svg)](https://pypi.org/project/pacbb/)
[![GitHub release](https://img.shields.io/github/release/yauhenii/pacbb.svg)](https://github.com/yauhenii/pacbb/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Archive](https://img.shields.io/badge/Archive-Yes-green.svg)](https://github.com/yauhenii/pacbb)

## About

The `pacbb` repository provides a collection of handy tools for PAC Bayes bounds evaluation. It is designed to simplify the work of PAC Bayes enthusiasts by offering various utilities and resources for easier implementation and experimentation.

We have prepared an arXiv paper that outlines the structure of ```pacbb```, demonstrates its usage, structure, and presents a series of experiments conducted using the toolkit.

## Links

- **Documentation**: [https://yauhenii.github.io/pacbb/core.html](https://yauhenii.github.io/pacbb/core.html)
- **PyPI**: [https://pypi.org/project/pacbb/](https://pypi.org/project/pacbb/)
- **Source Code**: [https://github.com/yauhenii/pacbb](https://github.com/yauhenii/pacbb)
- **Issues**: [https://github.com/yauhenii/pacbb/issues](https://github.com/yauhenii/pacbb/issues)
- **ArXiv**: [TODO](https://google.com)

## Installation

To install the `pacbb` package, use the following command:

```
pip install pacbb
```

## Example

For a complete example, please refer to the full script in [scripts/generic_train.py](https://github.com/yauhenii/pacbb/blob/main/scripts/generic_train.py).

Here is a part of this script showing how to convert a standard model to a Probabilistic Neural Network (ProbNN), which can be used for PAC Bayes boundaries calculation:

```python
from core.model import dnn_to_probnn
from core.distribution import GaussianVariable
from core.distribution.utils import from_random, from_zeros

# Initialize prior
prior_reference = from_zeros(model=model, 
                             rho=torch.log(torch.exp(torch.Tensor([sigma])) - 1), 
                             distribution=GaussianVariable, 
                             requires_grad=False)

prior = from_random(model=model, 
                    rho=torch.log(torch.exp(torch.Tensor([sigma])) - 1), 
                    distribution=GaussianVariable, 
                    requires_grad=True)

# Convert the model to ProbNN
dnn_to_probnn(model, prior, prior_reference)
```

Distributions creation and model conversion are explained in details in the arXive paper.

## Experiments

To run the experiments from the arXiv paper, follow these steps:

1. Clone the repository:

```
git clone https://github.com/Yauhenii/pacbb.git
```

2. Set up the environment:

```
conda create --prefix=./conda_env python=3.11 pip install -r requirements.txt
```

3. Create your desired experiment configuration:

```
./config
```

4. Run the configuration using the Python script directly:

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
python scripts/ivon_generic_train.py --config ./config/ivon_generic_configs/best_ivon.yaml
```

Alternatively, run multiple configuration files using a bash script wrapper:

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)" 
bash jobs/runnig_ivon_configs_in_the_folder.sh ./config/ivon_generic_configs/best_ivon.yaml
```


## Contribution

Contributions to `pacbb` are welcome! To contribute:

1. Fork the repository.
2. Create a new branch from `main`.
3. Make your changes.
4. Submit a pull request to the `main` branch.

Please use the following naming conventions for your branches:

- `feature/short-description` for new feature proposals.
- `bugfix/short-description` for bug fixes.
- `experiments/short-description` for changes related to the `scripts` module.

## Acknowledgments

Special thanks to **Vincent Fortuin** and **Alex Immer** for their supervision, support, and contributions to this project. Their guidance has been invaluable throughout the development of `pacbb`.


## Authors

* **Yauhenii** (Yauheni Mardan)

* **Misipuk** (Maksym Tretiakov)

