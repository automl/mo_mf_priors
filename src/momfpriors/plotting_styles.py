from __future__ import annotations

MARKERS = {
    "good-good": "o",      # Circle
    "good-bad": "v",       # Triangle Down
    "good-medium": "s",    # Square
    "good": "h",           # Hexagon
    "bad-good": "^",       # Triangle Up
    "bad-bad": "x",        # X
    "bad-medium": "d",     # Diamond
    "bad": "P",            # Plus (Filled)
    "medium-good": "p",    # Pentagon
    "medium-bad": "*",     # Star
    "medium-medium": "+",  # Plus
    "medium": "8",         # Octagon
}

COLORS = {
    "SMAC_ParEGO": "blue",
    "RandomSearch": "olive",
    "NepsRW": "lightseagreen",
    "NepsHyperbandRW": "darkgreen",
    "NepsHyperbandRW_w_continuations": "slategray",
    "Nevergrad_EvolutionStrategy": "orange",
    "MOMFBO": "purple",
    "MOMFBO_w_continuations": "darkpink",
    "NepsMOASHA": "salmon",
    "NepsMOASHA_w_continuations": "darkviolet",
    "RandomSearchWithPriors-good-good": "red",
    "RandomSearchWithPriors-good-bad": "teal",
    "RandomSearchWithPriors-good-medium": "deepskyblue",
    "RandomSearchWithPriors-good": "tomato",
    "RandomSearchWithPriors-bad-good": "darkcyan",
    "RandomSearchWithPriors-bad-bad": "mediumslateblue",
    "RandomSearchWithPriors-bad-medium": "crimson",
    "RandomSearchWithPriors-bad": "green",
    "RandomSearchWithPriors-medium-good": "darkorange",
    "RandomSearchWithPriors-medium-bad": "mediumturquoise",
    "RandomSearchWithPriors-medium-medium": "orchid",
    "RandomSearchWithPriors-medium": "sienna",
}

RC_PARAMS = {
    "text.usetex": False,  # True,
    # "pgf.texsystem": "pdflatex",
    # "pgf.rcfonts": False,
    # "font.family": "serif",
    # "font.serif": [],
    # "font.sans-serif": [],
    # "font.monospace": [],
    "font.size": "10.90",
    "legend.fontsize": "9.90",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.title_fontsize": "small",
    "lines.linewidth": 1,
    "patch.linewidth": 1,
    "axes.linewidth": 1,
    "lines.markersize": 4,
    # "bottomlabel.weight": "normal",
    # "toplabel.weight": "normal",
    # "leftlabel.weight": "normal",
    # "tick.labelweight": "normal",
    # "title.weight": "normal",
    # "pgf.preamble": r"""
    #    \usepackage[T1]{fontenc}
    #    \usepackage[utf8x]{inputenc}
    #    \usepackage{microtype}
    # """,
}

XTICKS = {
    (1, 25): [1, 5, 10, 15, 20, 25],
    (1, 5): [1, 2, 3, 4, 5],
    (1, 10): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (1, 20): [1, 4, 8, 12, 16, 20],
}

reference_points_dict = {

    # PD1
    "pd1-cifar100-wide_resnet-2048": {"valid_error_rate": 1, "train_cost": 100},
    "pd1-imagenet-resnet-512": {"valid_error_rate": 1, "train_cost": 5000},
    "pd1-lm1b-transformer-2048": {"valid_error_rate": 1, "train_cost": 1000},
    "pd1-translate_wmt-xformer_translate-64": {"valid_error_rate": 1, "train_cost": 20000},

    # JAHSBench
    "jahs-CIFAR10": {"valid_acc": 0, "runtime": 200000},
    "jahs-ColorectalHistology": {"valid_acc": 0, "runtime": 200000},
    "jahs-FashionMNIST": {"valid_acc": 0, "runtime": 200000},

    # MOMFPark
    "MOMFPark": {"value1": 1, "value2": 1},

    # YAHPO-LCBench
    "yahpo-lcbench-126026": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-167190": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-168330": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 20000
    },
    "yahpo-lcbench-168910": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189906": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 1000
    },
    "yahpo-lcbench-3945": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-7593": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-34539": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126025": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126029": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-146212": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167104": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167149": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167152": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167161": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167168": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167181": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167184": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167185": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167200": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167201": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168329": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168331": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168335": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168868": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189354": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189862": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189865": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189866": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189873": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189905": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189909": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    }
}
