import hydra
from omegaconf import DictConfig

from core.dataset import DatasetHandler


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    print(config)
    dataset_handler = DatasetHandler(dataset_name=config["experiment_settings"]["dataset_name"],
                                     dataset_path=config["experiment_settings"]["dataset_path"],
                                     split_strategy_name=config["experiment_settings"]["split_strategy_name"])
    dataset_handler.load_and_split_dataset()



if __name__ == '__main__':
    main()
