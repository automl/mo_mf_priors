from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momfpriors.constants import DEFAULT_RESULTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
) -> None:
    """Function to plot the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    logger.info(f"\nPlots for runs on benchmark: {benchmark}")
    plt.figure(figsize=(10, 10))
    for instance, instance_data in data.items():
        logger.info(f"Plotting pareto front for {instance}")
        results = instance_data["results"]
        plot_title = instance_data["plot_title"]
        keys = list(results[0].keys())
        assert len(keys) == 2, "Can only plot pareto front for 2D cost space."  # noqa: PLR2004
        pareto = pareto_front(results)
        pareto = np.array([list(res.values()) for res in results])[pareto]
        pareto = pareto[pareto[:, 0].argsort()]
        logger.info(f"Plotting pareto front for {plot_title}")
        plt.plot(
            pareto[:, 0],
            pareto[:, 1],
            marker="o",
            markersize=7,
            linewidth=1,
        )
    plt.grid()
    plt.legend(list(data.keys()))
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    if len(data) > 1:
        plot_title = f"Pareto front for Optimizers on {benchmark} benchmark"
    plt.title(plot_title)
    save_dir = exp_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{plot_title.replace(' ', '_')}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        required=True,
        help="Main experiment directory containing the runs to plot."
    )
    args = parser.parse_args()
    exp_dir: Path = DEFAULT_RESULTS_DIR / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}


    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")


    for benchmark in benchmarks_in_dir:
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue 
            _df = pd.read_parquet(file)
            instance = _df["optimizer"][0] + "_" + _df["optimizer_hps"][0]
            agg_dict[instance] = {
                "results": _df["results"],
                "optimizer": _df["optimizer"][0],
                "opt_hps": _df["optimizer_hps"][0],
                "plot_title": (
                    f"{_df['optimizer'][0]}_"
                    f"{_df['optimizer_hps'][0]}"
                    f" on {benchmark}"
                ),
            }
        plot_pareto(
            data=agg_dict,
            exp_dir=exp_dir,
            benchmark=benchmark,
        )
