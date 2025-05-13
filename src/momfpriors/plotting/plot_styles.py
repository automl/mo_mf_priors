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
    "RandomSearch": "sienna",
    "NepsRW": "lightseagreen",
    "NepsHyperbandRW": "gainsboro",
    "NepsHyperbandRW_w_continuations": "darkslategray",
    "Nevergrad_EvolutionStrategy": "lime",
    "MOMFBO": "purple",
    "MOMFBO_w_continuations": "magenta",
    "NepsMOASHA": "salmon",
    "NepsMOASHA_w_continuations": "darkviolet",
    "NepsPriorMOASHA-good-good": "thistle",
    "NepsPriorMOASHA-good-good_w_continuations": "darkgreen",
    "NepsPriorMOASHA-bad-bad": "crimson",
    "NepsPriorMOASHA-bad-bad_w_continuations": "hotpink",
    "NepsMOPriorband-good-good": "pink",
    "NepsMOPriorband-good-good_w_continuations": "cornflowerblue",
    "NepsMOPriorband-bad-bad": "lightcoral",
    "NepsMOPriorband-bad-bad_w_continuations": "olivedrab",
    "NepsPiBORW-good-good": "black",
    "NepsPiBORW-bad-bad": "darkcyan",
    "NepsMOASHABO": "paleturquoise",
    "NepsMOASHABO_w_continuations": "darkblue",
    "NepsMOASHAPiBORW-good-good": "saddlebrown",
    "NepsMOASHAPiBORW-good-good_w_continuations": "goldenrod",
    "NepsMOASHAPiBORW-bad-bad": "lightsteelblue",
    "NepsMOASHAPiBORW-bad-bad_w_continuations": "darkkhaki",
    "RandomSearchWithPriors-all": "darkseagreen",
    "RandomSearchWithPriors-good-good": "red",
    "RandomSearchWithPriors-good-bad": "teal",
    "RandomSearchWithPriors-good-medium": "deepskyblue",
    "RandomSearchWithPriors-good-None": "tomato",
    "RandomSearchWithPriors-None-good": "lightblue",
    "RandomSearchWithPriors-bad-good": "green",
    "RandomSearchWithPriors-bad-bad": "darkorange",
    "RandomSearchWithPriors-bad-medium": "crimson",
    "RandomSearchWithPriors-bad-None": "peru",
    "RandomSearchWithPriors-None-bad": "lightcoral",
    "RandomSearchWithPriors-medium-good": "darkorange",
    "RandomSearchWithPriors-medium-bad": "mediumturquoise",
    "RandomSearchWithPriors-medium-medium": "orchid",
    "RandomSearchWithPriors-medium-None": "darkkhaki",
    "RandomSearchWithPriors-None-medium": "lightpink",
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
        6 : 3,
        8 : 4,
        9 : 6,
        12: 8,
    },
    "ylabel_i_inc": {
        2 : 0,
        3 : 0,
        4 : 2,
        6 : 3,
        8 : 4,
        9 : 3,
        12: 4,
    }
}
