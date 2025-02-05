from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from momfpriors.constants import DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


with (DEFAULT_ROOT_DIR / "configs" / "plotting_styles.yaml").open("r") as f:
    style_dict = yaml.safe_load(f)


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


def _get_style(opt: str, prior_annot: str) -> tuple[str, str]:
    """Function to get the plotting style for a given instance."""
    color = style_dict["colors"].get(opt, "black")
    annotations = [a.split("=")[-1] for a in prior_annot.split(",")]
    annotations = "-".join(annotations)
    marker = style_dict["markers"].get(annotations, "o")
    return marker, color


def plot_pareto(
    data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
    *,
    is_single_opt: bool = False,
) -> None:
    """Function to plot the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    logger.info(f"\nPlots for runs on benchmark: {benchmark}")
    plt.figure(figsize=(10, 10))
    for _, instance_data in data.items():
        results = instance_data["results"]
        plot_title = instance_data["plot_title"]
        keys = list(results[0].keys())
        assert len(keys) == 2, "Can only plot pareto front for 2D cost space."  # noqa: PLR2004
        pareto = pareto_front(results)
        pareto = np.array([list(res.values()) for res in results])[pareto]
        pareto = pareto[pareto[:, 0].argsort()]
        logger.info(f"Plotting pareto front for {plot_title}")
        marker, color = _get_style(
            instance_data["optimizer"],
            instance_data["prior_annotations"],
        )
        plt.step(
            pareto[:, 0],
            pareto[:, 1],
            where="post",
            marker=marker,
            color=color if not is_single_opt else None,
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
            instance = (
                _df["optimizer"][0] + ";" + _df["optimizer_hyperparameters"][0] + ";" +
                _df["prior_annotations"][0]
            )
            agg_dict[instance] = {
                "results": _df["results"],
                "optimizer": _df["optimizer"][0],
                "prior_annotations": _df["prior_annotations"][0],
                "plot_title": (
                    f"{instance}"
                    f" on {benchmark}"
                ),
            }
        is_single_opt = False
        if len({agg_dict[instance]["optimizer"] for instance in agg_dict}) == 1:
            is_single_opt = True
        plot_pareto(
            data=agg_dict,
            exp_dir=exp_dir,
            benchmark=benchmark,
            is_single_opt=is_single_opt,
        )
