from core.objective import AbstractObjective, BBBObjective
from app.utils import AbstractFactory


class ObjectiveFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("bbb", BBBObjective)

    def create(self, objective_name: str, *args, **kwargs) -> AbstractObjective:
        creator = self._creators.get(objective_name)
        if not creator:
            raise ValueError(f"Invalid objective: {objective_name}")
        return creator(*args, **kwargs)
