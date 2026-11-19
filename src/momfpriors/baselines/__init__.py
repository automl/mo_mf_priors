from __future__ import annotations

from momfpriors.baselines.ablation import NepsMOASHA_RS
from momfpriors.baselines.momfbo import MOMFBO_Optimizer
from momfpriors.baselines.neps_optimizers import (
    NepsEtaPriorMFPriMO,
    NepsEtaPriorPriMO,
    NepsHyperbandRW,
    NepsInitPiBORW,
    NepsMFPriMO,
    NepsMOASHA,
    NepsMOASHABO,
    NepsMOBO,
    NepsMOHyperband,
    NepsMOPriorband,
    NepsNoInitPriMO,
    NepsPiBORW,
    NepsPiBORW_eps,
    NepsPiBORW_legacy,
    NepsPriMO,
    NepsPriMO_EHVI,
    NepsPriMO_HV,
    NepsPriMO_Init10,
    NepsPriMO_LCB,
    NepsPriMO_noEPS,
    NepsPriMO_TS,
    NepsPriorMOASHA,
    NepsPriorRSMOASHA,
    NepsRW,
)
from momfpriors.baselines.nevergrad import yield_nevergrad_optimizers
from momfpriors.baselines.optuna import OptunaOptimizer
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
    NepsMOBO.name: NepsMOBO,
    NepsPiBORW.name: NepsPiBORW,
    NepsPiBORW_legacy.name: NepsPiBORW_legacy,
    NepsPriMO.name: NepsPriMO,
    NepsMOASHA_RS.name: NepsMOASHA_RS,
    NepsPriorRSMOASHA.name: NepsPriorRSMOASHA,
    NepsNoInitPriMO.name: NepsNoInitPriMO,
    NepsInitPiBORW.name: NepsInitPiBORW,
    NepsEtaPriorMFPriMO.name: NepsEtaPriorMFPriMO,
    NepsEtaPriorPriMO.name: NepsEtaPriorPriMO,
    NepsMFPriMO.name: NepsMFPriMO,
    NepsPriMO_Init10.name: NepsPriMO_Init10,
    OptunaOptimizer.name: OptunaOptimizer,
    NepsPriMO_HV.name: NepsPriMO_HV,
    NepsPriMO_TS.name: NepsPriMO_TS,
    NepsPriMO_LCB.name: NepsPriMO_LCB,
    NepsPriMO_noEPS.name: NepsPriMO_noEPS,
    NepsPiBORW_eps.name: NepsPiBORW_eps,
    NepsPriMO_EHVI.name: NepsPriMO_EHVI,
}

for opt in yield_nevergrad_optimizers():
    OPTIMIZERS[opt.name] = opt

__all__ = ["OPTIMIZERS"]
