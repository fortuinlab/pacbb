from app.dataset.split_strategy import AbstractSplitStrategy, PBPSplitStrategy, FaultySplitStrategy
from app.utils import AbstractFactory


class SplitStrategyFactory(AbstractFactory):
    """
    Factory class for creating split strategy objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("pbp", PBPSplitStrategy)
        self.register_creator("faulty_pbp", FaultySplitStrategy)

    def create(self, split_strategy_name: str) -> AbstractSplitStrategy:
        """
        Create a split strategy object.

        Args:
            split_strategy_name (str): Name of the split strategy.

        Returns:
            AbstractSplitStrategy: Created split strategy object.

        Raises:
            ValueError: If an invalid split strategy name is provided.
        """
        creator = self._creators.get(split_strategy_name)
        if not creator:
            raise ValueError(f"Invalid split strategy: {split_strategy_name}")
        return creator(split_strategy_name)
