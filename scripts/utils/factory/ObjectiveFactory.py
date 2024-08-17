from core.objective import (AbstractObjective,
                            FClassicObjective,
                            McAllisterObjective,
                            FQuadObjective,
                            BBBObjective,
                            TolstikhinObjective,
                            NaiveIWAEObjective,
                            IWAEObjective)

from scripts.utils.factory import AbstractFactory


class ObjectiveFactory(AbstractFactory[AbstractObjective]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("bbb", BBBObjective)
        self.register_creator("fclassic", FClassicObjective)
        self.register_creator("fquad", FQuadObjective)
        self.register_creator("mcallister", McAllisterObjective)
        self.register_creator("tolstikhin", TolstikhinObjective)
        self.register_creator("naive_iwae", NaiveIWAEObjective)
        self.register_creator("iwae", IWAEObjective)
