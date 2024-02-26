import hydra
import torch
from omegaconf import DictConfig

from core.pipeline.training import TrainingPipelineFactory
from core.pipeline.evaluaiton import EvaluationPipelineFactory


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    if __name__ == '__main__':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_pipeline = TrainingPipelineFactory().create(config['training_pipeline']['training_pipeline_name'])
        training_pipeline.train(model_config=config['training_pipeline']['model'],
                                dataset_config=config['dataset'],
                                split_strategy_config=config['split_strategy'],
                                device=device)

        evaluation_pipeline = EvaluationPipelineFactory().create(config['evaluation_pipeline']['evaluation_pipeline_name'])
        evaluation_pipeline.evaluate()


if __name__ == '__main__':
    main()
