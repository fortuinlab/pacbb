##  About

This repository was developed to collect handy tools for PAC Bayes bounds evaluation and 
hence make a life of PAC Bayes enthusiasts easier.

This repository is structured in the following way: `core` module is used a


## Package

### Docs

Documentation  is available here
https://yauhenii.github.io/pacbb/core.html

### Installation


## Experiments 

To run the experiments, first set up the environment
```
conda create --prefix=./conda_env python=3.11
pip install -r requirements.txt
```
Create desired experiments configuration
```
./config
```
Run configuration using python script directly
```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/ivon_generic_train.py --config ./config/ivon_generic_configs/best_ivon.yaml
```

Or run multiple config files using bash script wrapping

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
bash jobs/runnig_ivon_configs_in_the_folder.sh ./config/ivon_generic_configs/best_ivon.yaml
```

## Contribution

Everyone is welcome to contribute to the `pacbb`. 
Just make a brunch from the `main`, make your changes, and create a pull request into `main`.
Please name  your branches as `feature/short-description`  for feature proposal, 
`bugfix/short-description` for bug fixes, 
and `experiments/short-description` for proposal to the `scripts` module.


