from __future__ import annotations

from momfpriors.baselines.ablation import NepsMOASHA_RS
from momfpriors.baselines.momfbo import MOMFBO_Optimizer
from momfpriors.baselines.neps_optimizers import (
    NepsHyperbandRW,
    NepsMOASHA,
    NepsMOASHABO,
    NepsPriMO,
    NepsMOHyperband,
    NepsMOPriorband,
    NepsPiBORW,
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
    NepsMOPriorband.name: NepsMOPriorband,
    NepsMOASHABO.name: NepsMOASHABO,
    NepsPiBORW.name: NepsPiBORW,
    NepsPriMO.name: NepsPriMO,
    NepsMOASHA_RS.name: NepsMOASHA_RS,
}

for opt in yield_nevergrad_optimizers():
    OPTIMIZERS[opt.name] = opt

__all__ = ["OPTIMIZERS"]
