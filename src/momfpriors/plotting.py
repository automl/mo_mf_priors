from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momfpriors.constants import DEFAULT_RESULTS_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pareto_front(
    costs: pd.Series,
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
        of Results, i.e., Mapping[str, float] objects.
    """
    costs: np.array = np.array([list(cost.values()) for cost in costs])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto

def plot_pareto(
    results: pd.Series,
    run_dir: Path,
    plot_title: str,
) -> None:
    """Function to plot the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    keys = list(results[0].keys())
    assert len(keys) == 2, "Can only plot pareto front for 2D cost space."  # noqa: PLR2004
    pareto = pareto_front(results)
    pareto = np.array([list(res.values()) for res in results])[pareto]
    logger.info(f"Plotting pareto front for {plot_title}")
    plt.scatter(pareto[:, 0], pareto[:, 1])
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.title(plot_title)
    save_dir = run_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "pareto_front.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        required=True,
        help="Main experiment directory containing the runs to plot."
    )
    args = parser.parse_args()
    exp_dir = DEFAULT_RESULTS_DIR / args.exp_dir

    for run_dir in exp_dir.iterdir():
        run = exp_dir / run_dir
        for file in run.iterdir():
            if file.suffix == ".parquet":
                _df = pd.read_parquet(file)
                results = _df["results"]
                benchmark = _df["benchmark"][0]
                optimizer = _df["optimizer"][0]
                opt_hps = _df["optimizer_hps"][0]
                plt_title = f"{optimizer}_{opt_hps} on {benchmark}"
                plot_pareto(
                    results=results,
                    run_dir=run_dir,
                    plot_title=plt_title,
                )
