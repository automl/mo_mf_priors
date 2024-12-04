from __future__ import annotations

from momfpriors.baselines.neps_optimizers import NepsOptimizer
from momfpriors.baselines.random_search import RandomSearch
from momfpriors.baselines.smac_parego import SMAC_ParEGO

OPTIMIZERS = {
    RandomSearch.name: RandomSearch,
    NepsOptimizer.name: NepsOptimizer,
    SMAC_ParEGO.name: SMAC_ParEGO,
}

__all__ = ["OPTIMIZERS"]
