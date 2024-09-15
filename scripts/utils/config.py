import yaml


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


from datetime import datetime


def get_wandb_name(config_path: str):
    prefix = './config/'
    postfix = '.yaml'

    # Remove prefix
    if config_path.startswith(prefix):
        config_path = config_path[len(prefix):]

    # Remove postfix
    if config_path.endswith(postfix):
        config_path = config_path[:-len(postfix)]

    # Add timestamp
    timestamp = datetime.now().strftime("%d%m_%H%M")

    return f"{config_path}_{timestamp}"

