from __future__ import annotations

from momfpriors.baselines.neps_optimizers import NepsOptimizer
from momfpriors.baselines.random_search import RandomSearch

OPTIMIZERS = {
    RandomSearch.name: RandomSearch,
    NepsOptimizer.name: NepsOptimizer,
}

__all__ = ["OPTIMIZERS"]
