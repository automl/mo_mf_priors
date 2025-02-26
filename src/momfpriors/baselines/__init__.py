from __future__ import annotations

from momfpriors.baselines.neps_optimizers import NepsRW
from momfpriors.baselines.random_search import RandomSearch, RandomSearchWithPriors
from momfpriors.baselines.smac_parego import SMAC_ParEGO

OPTIMIZERS = {
    RandomSearch.name: RandomSearch,
    RandomSearchWithPriors.name: RandomSearchWithPriors,
    NepsRW.name: NepsRW,
    SMAC_ParEGO.name: SMAC_ParEGO,
}

__all__ = ["OPTIMIZERS"]
