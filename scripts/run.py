import hydra
from omegaconf import DictConfig

from core.dataset import DatasetHandler


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    print(config)
    dataset_handler = DatasetHandler(dataset_config=config["experiment_settings"]["dataset"],
                                     split_strategy_config=config["experiment_settings"]["split_strategy"])
    dataset_handler.load_and_split_dataset()


if __name__ == '__main__':
    main()
