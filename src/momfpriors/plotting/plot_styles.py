from __future__ import annotations

MARKERS = {
    "good-good": "o",      # Circle
    "good-bad": "v",       # Triangle Down
    "good-medium": ".",    # dot
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
    # Optimizers without Priors
    "SMAC_ParEGO": "goldenrod",
    "RandomSearch": "peru",
    "NepsRW": "lightseagreen",
    "Nevergrad_EvolutionStrategy": "lime",

    "NepsHyperbandRW": "darkslategray",
    "NepsHyperbandRW_w_continuations": "darkslategray",

    "MOMFBO": "magenta",
    "MOMFBO_w_continuations": "magenta",

    "NepsMOASHA": "darkviolet",
    "NepsMOASHA_w_continuations": "darkviolet",

    "NepsMOASHA_RS": "violet",
    "NepsMOASHA_RS_w_continuations": "violet",

    "NepsMOASHABO": "darkblue",
    "NepsMOASHABO_w_continuations": "darkblue",

    # Optimizers with Priors

    # NepsPriorMOASHA
    "NepsPriorMOASHA": "darkgreen",
    "NepsPriorMOASHA_w_continuations": "darkgreen",

    # NepsMOPriorband
    "NepsMOPriorband": "cornflowerblue",
    "NepsMOPriorband_w_continuations": "cornflowerblue",

    # NepsPiBORW
    "NepsPiBORW": "olive",

    # NepsMOASHAPiBORW
    "NepsMOASHAPiBORW": "crimson",
    "NepsMOASHAPiBORW_w_continuations": "crimson",

    # RandomSearchWithPriors
    "RandomSearchWithPriors": "darkorange",
}

RC_PARAMS = {
    "text.usetex": False,  # True,
    # "pgf.texsystem": "pdflatex",
    # "pgf.rcfonts": False,
    # "font.family": "serif",
    # "font.serif": [],
    # "font.sans-serif": [],
    # "font.monospace": [],
    "font.size": "20",
    "legend.fontsize": "9.90",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.title_fontsize": "small",
    "lines.linewidth": 1,
    "patch.linewidth": 1,
    "axes.linewidth": 1,
    "lines.markersize": 4,
    "axes.titlesize": 20,
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
    (1, 15): [1, 3, 6, 9, 12, 15],
    (1, 20): [1, 4, 8, 12, 16, 20],
}


other_fig_params = {
    "fig_size": (20, 15),
    "ovrank_xsize": 7,
    "ovrank_ysize": 7,
    "n_rows_cols": {
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        8: (2, 4),
        9: (3, 3),
        12: (3, 4),
    },
    "tight_layout_pads": {
        "pad": 0,
        "h_pad": 0.5,
    },
    "bbox_to_anchor": (0.5, -0.08),
    "legend_fontsize": 18,
    "xylabel_fontsize": 20,
    "xlabel_start_i": {
        2 : 0,
        3 : 0,
        4 : 2,
        5 : 3,
        6 : 3,
        8 : 4,
        9 : 6,
        12: 8,
    },
    "ylabel_i_inc": {
        2 : 0,
        3 : 0,
        4 : 2,
        5 : 3,
        6 : 3,
        8 : 4,
        9 : 3,
        12: 4,
    },
    "multi_fig_leg_cols": {
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
        10: 5,
        11: 4,
        12: 4,
        13: 5,
        14: 5,
        15: 5,
        16: 4,
    },
    "single_fig_leg_cols": {
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 3,
        8: 2,
        9: 3,
        10: 2,
        11: 3,
        12: 2,
        13: 5,
    }
}
