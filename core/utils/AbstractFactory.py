from abc import ABC, abstractmethod
from typing import Type, TypeVar

T = TypeVar("T")  # can be bounded


class AbstractFactory(ABC):
    """
    Abstract Factory class providing a blueprint for factory implementations.
    """

    def __init__(self) -> None:
        self._creators: dict[str, Type[T]] = {}

    def register_creator(self, name: str, cls: Type[T]) -> None:
        """
        Register a creator class with the factory.

        Args:
            name (str): Name of the creator class.
            cls (Type[T]): Creator class to register.
        """
        self._creators[name] = cls

    @abstractmethod
    def create(self, *args, **kwargs) -> T:
        """
        Abstract method to create an object.

        Returns:
            T: Created object.
        """
        pass
