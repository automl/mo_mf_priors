from __future__ import annotations

from momfpriors.baselines.momfbo import MOMFBO_Optimizer
from momfpriors.baselines.neps_optimizers import (
    NepsHyperbandRW,
    NepsMOASHA,
    NepsMOHyperband,
    NepsPriorMOASHA,
    NepsRW,
)
from momfpriors.baselines.nevergrad import yield_nevergrad_optimizers
from momfpriors.baselines.random_search import RandomSearch, RandomSearchWithPriors
from momfpriors.baselines.smac_parego import SMAC_ParEGO

OPTIMIZERS = {
    RandomSearch.name: RandomSearch,
    RandomSearchWithPriors.name: RandomSearchWithPriors,
    NepsRW.name: NepsRW,
    NepsHyperbandRW.name: NepsHyperbandRW,
    SMAC_ParEGO.name: SMAC_ParEGO,
    MOMFBO_Optimizer.name: MOMFBO_Optimizer,
    NepsMOASHA.name: NepsMOASHA,
    NepsPriorMOASHA.name: NepsPriorMOASHA,
    NepsMOHyperband.name: NepsMOHyperband,
}

for opt in yield_nevergrad_optimizers():
    OPTIMIZERS[opt.name] = opt

__all__ = ["OPTIMIZERS"]
