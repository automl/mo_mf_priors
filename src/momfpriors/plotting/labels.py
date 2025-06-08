from __future__ import annotations

INTRO_LABELS = {
    "NepsRW": "BO+RW",
    "RandomSearchWithPriors": "RS+Prior",
    "NepsMOASHAPiBORW": "Our Method",
    "NepsMOASHA": "MOASHA",
}



LABELS_1 = {

    "NepsMOASHAPiBORW": "Our Method",

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "Random Search",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "MOASHA-BO+RW",

    "MOMFBO": "MOMF-BO",

    "NepsPriorMOASHA": "MOASHA+Prior",

    "NepsMOPriorband": "MO-Priorband",

    "NepsPiBORW": "πBO+RW",

    "RandomSearchWithPriors": "RS+Prior",
}


ABLATION_LABELS = {

    "NepsRW": "Model",

    "NepsMOASHA_RS": "MF Initial Design + RS",

    "NepsMOASHAPiBORW": "MF initial design + Priors + Model",

    "NepsMOASHABO": "MF initial design + Model",

    "NepsPiBORW": "Priors + Model",
}

HP_LABELS = {
    "initial_design_size": "init",
}






#################
# NOT IN USE
#################

LABELS_99 = {

    # Our Optimizer

    "NepsMOASHAPiBORW-all": "Our Method(all)",
    "NepsMOASHAPiBORW-good-good": "Our Method(good-good)",
    "NepsMOASHAPiBORW-bad-bad": "Our Method(bad-bad)",
    "NepsMOASHAPiBORW-bad-good": "Our Method(bad-good)",

    # Optimizers without Priors

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "Random Search",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "MOASHA-BO+RW",

    "MOMFBO": "MOMF-BO",

    # Optimizers with Priors

    "NepsPriorMOASHA-all": "MOASHA+Prior(all)",
    "NepsPriorMOASHA-good-good": "MOASHA+Prior(good-good)",
    "NepsPriorMOASHA-bad-bad": "MOASHA+Prior(bad-bad)",
    "NepsPriorMOASHA-bad-good": "MOASHA+Prior(bad-good)",

    "NepsMOPriorband-all": "MO-Priorband(all)",
    "NepsMOPriorband-good-good": "MO-Priorband(good-good)",
    "NepsMOPriorband-bad-bad": "MO-Priorband(bad-bad)",
    "NepsMOPriorband-bad-good": "MO-Priorband(bad-good)",

    "NepsPiBORW-all": "πBO+RW(all)",
    "NepsPiBORW-good-good": "πBO+RW(good-good)",
    "NepsPiBORW-bad-bad": "πBO+RW(bad-bad)",
    "NepsPiBORW-bad-good": "πBO+RW(bad-good)",

    "RandomSearchWithPriors-all": "RS+Prior(all)",
    "RandomSearchWithPriors-good-good": "RS+Prior(good-good)",
    "RandomSearchWithPriors-good-bad": "RS+Prior(good-bad)",
    "RandomSearchWithPriors-good-medium": "RS+Prior(good-medium)",
    "RandomSearchWithPriors-good-None": "RS+Prior(good-None)",
    "RandomSearchWithPriors-None-good": "RS+Prior(None-good)",
    "RandomSearchWithPriors-bad-good": "RS+Prior(bad-good)",
    "RandomSearchWithPriors-bad-bad": "RS+Prior(bad-bad)",
    "RandomSearchWithPriors-bad-medium": "RS+Prior(bad-medium)",
    "RandomSearchWithPriors-bad-None": "RS+Prior(bad-None)",
    "RandomSearchWithPriors-None-bad": "RS+Prior(None-bad)",
    "RandomSearchWithPriors-medium-good": "RS+Prior(medium-good)",
    "RandomSearchWithPriors-medium-bad": "RS+Prior(medium-bad)",
    "RandomSearchWithPriors-medium-medium": "RS+Prior(medium-medium)",
    "RandomSearchWithPriors-medium-None": "RS+Prior(medium-None)",
    "RandomSearchWithPriors-None-medium": "RS+Prior(None-medium)",
}