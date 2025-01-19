"""
The `objective` subpackage holds various PAC-Bayes training objectives.

Included classes:
  - `AbstractObjective` (interface)
  - `BBBObjective`, `FClassicObjective`, `FQuadObjective`, `McAllisterObjective`,
    and `TolstikhinObjective`.

Each objective combines empirical loss with a KL term (and additional factors)
to minimize a PAC-Bayes bound during training.
"""

from core.objective.AbstractObjective import AbstractObjective
from core.objective.BBBObjective import BBBObjective
from core.objective.FQuadObjective import FQuadObjective
from core.objective.FClassicObjective import FClassicObjective
from core.objective.McAllisterObjective import McAllisterObjective
from core.objective.TolstikhinObjective import TolstikhinObjective
