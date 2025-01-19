import yaml
import os
import logging
from datetime import datetime


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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


def setup_logging(config_path: str, logging_dir: str = 'logs'):
    relative_log_path = config_path.replace('./config/', '').replace('.yaml', '.log')

    log_file_path = os.path.join(logging_dir, relative_log_path)

    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging to write to both file and console, flush real-time
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging setup complete. Logs will be saved to {log_file_path}")

    return log_file_path
