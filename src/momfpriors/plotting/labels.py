from __future__ import annotations

INTRO_LABELS = {
    "SMAC_ParEGO": "ParEGO",
    "NepsRW": "BO+RW",
    "RandomSearchWithPriors": "RS+Prior",
    "NepsPriMO": "PriMO",
    "NepsMOASHA": "MOASHA",
    "Nevergrad_EvolutionStrategy": "NSGA-II",
}



LABELS_1 = {

    "NepsPriMO": "PriMO",

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "Random Search",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "NepsMOASHABO": "PriMO w/o Priors",

    "MOMFBO": "MOMF-BO",

    "NepsPriorMOASHA": "MOASHA+Prior",

    "NepsMOPriorband": "MO-Priorband",

    "NepsPiBORW": "πBO+RW with NePS prior decay",

    "NepsPiBORW_legacy": "πBO+RW",

    "RandomSearchWithPriors": "RS+Prior",

    "NepsPriorRSMOASHA": "MOASHA+Prior(50%)",

    "Optuna": "MO-TPE",

    "NepsMOBO": "qLogNEHVI",
}

EPS_STUDY_LABELS = {
    **LABELS_1,
    "Optuna": "Strongest baseline",
}

PIBO_COMP_LABELS = {

    "NepsPiBORW": "πBO+RW with NePS prior decay",

    "NepsPiBORW_legacy": "πBO+RW with original prior decay",

}


ABLATION_LABELS = {

    "NepsPriMO_HV": "PriMO+HV",

    "NepsPriMO_TS": "PriMO+TS",

    "NepsPriMO_LCB": "PriMO+LCB",

    "NepsPriMO_EHVI": "PriMO+EHVI",

    "NepsNoInitPriMO": "PriMO w/o initial design",

    "NepsMOASHA_RS": "PriMO w/o Priors and BO",

    "NepsPriMO": "PriMO",

    "NepsPriMO_noEPS": "PriMO's prior decay w/o ϵ-greedy",

    "NepsPiBORW_eps": "πBO's prior decay + ϵ-greedy",

    "NepsMOASHABO": "PriMO w/o Priors",

    "NepsInitPiBORW": "PriMO w/o ϵ-BO",

    "NepsEtaPriorPriMO": "PriMO + 1/η Priors in initial design",

    "NepsMFPriMO": "MO-ASHA + PriMO Sampler",

    "NepsEtaPriorMFPriMO": "MO-ASHA + 1/η Priors + PriMO Sampler",

    "Optuna": "Strongest baseline",
}

ABLATION_INIT_LABELS = {

    "NepsPriMO": "init=5",
    "NepsNoInitPriMO": "init=0",
    "NepsPriMO_Init10": "init=10",
}

HP_LABELS = {
    "initial_design_size": "init",
    "beta": r"$\beta$",
}

SIG_LABELS = {

    "NepsPriMO": "PriMO",

    "SMAC_ParEGO": "ParEGO",
    "RandomSearch": "RS",
    "NepsRW": "BO+RW",
    "Nevergrad_EvolutionStrategy": "NSGA-II",

    "NepsHyperbandRW": "HB+RW",

    "NepsMOASHA": "MOASHA",

    "NepsMOASHA_RS": "MOASHA+RS",

    "MOMFBO": "MOMF-BO",

    "NepsPriorMOASHA": "MOASHA+Prior",

    "NepsMOPriorband": "MO-PB",

    "NepsPiBORW": "πBO+RW",

    "NepsPiBORW_legacy": "πBO+RW",

    "RandomSearchWithPriors": "RS+Prior",

    "Optuna": "MO-TPE",

    "NepsMOBO": "qLogNEHVI",
}

SO_LABELS = {

    "NepsHyperband": "HyperBand",
    "NepsBO": "BO",
    "NepsPriorband": "Priorband",
    "NepsPriorbandBO": "Priorband+BO",
    "NepsMFBO": "PriMO",
    "NepsPiBO": "πBO",
}


GENERAL_HPOSUITE = {
    "NepsBO": "Neps BO",
    "NepsHyperband": "Neps HyperBand",
    "NepsPriorband": "Neps Priorband",
    "SMAC_BO": "SMAC BO",
    "RandomSearch": "Random Search",
    "Scikit_Optimize": "Skopt BO",
    "Optuna": "Optuna TPE",
}

