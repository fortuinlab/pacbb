from typing import Callable

from core.loss import nll_loss, scaled_nll_loss,  zero_one_loss

from scripts.utils.factory import AbstractFactory


class LossFactory(AbstractFactory[Callable]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("nll_loss", nll_loss)
        self.register_creator("scaled_nll_loss", scaled_nll_loss)
        self.register_creator("01_loss", zero_one_loss)

    def create(self, name: str, *args, **kwargs) -> Callable:
        """
         A method to create an object.

        Args:
            name (str): Name of the creator class.

        Returns:
            T: Created object.
        """
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Invalid creator: {name}")
        return creator
