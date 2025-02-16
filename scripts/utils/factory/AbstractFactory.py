from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T")


class AbstractFactory(Generic[T], ABC):
    """
    Abstract Factory class providing a blueprint for factory implementations.
    """

    def __init__(self) -> None:
        self._creators: dict[str, type[T]] = {}

    def register_creator(self, name: str, cls: type[T]) -> None:
        """
        Register a creator class with the factory.

        Args:
            name (str): Name of the creator class.
            cls (Type[T]): Creator class to register.
        """
        self._creators[name] = cls

    def create(self, name: str, *args, **kwargs) -> T:
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
        return creator(*args, **kwargs)
