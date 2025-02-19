from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt
import yaml
from pymoo.indicators.hv import Hypervolume

from momfpriors.constants import DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


with (DEFAULT_ROOT_DIR / "configs" / "plotting_styles.yaml").open("r") as f:
    style_dict = yaml.safe_load(f)


def pareto_front(
    costs: pd.Series | list[Mapping[str, float]]
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


def create_plots(  # noqa: PLR0915
    agg_data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
    budget: int,
    *,
    is_single_opt: bool = False,
    pareto_seeds: bool = False,
) -> None:
    """Function to plot the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    logger.info(f"\nPlots for runs on benchmark: {benchmark}")
    reference_point = np.array([-np.inf, -np.inf])
    plt.figure(1, figsize=(7, 5))
    plt.figure(2, figsize=(7, 5))
    if len(agg_data) > 1:
        plot_title = f"Optimizers on \n{benchmark}"
    for instance, instance_data in agg_data.items():
        logger.info(f"\nPlots for instance: {instance}")
        seed_hv_dict = {}
        for seed, data in instance_data.items():
            results = data["results"]
            plot_title = data["plot_title"]
            keys = list(results[0].keys())
            assert len(keys) == 2, "Can only plot pareto front for 2D cost space."  # noqa: PLR2004
            acc_costs = []
            pareto = None
            hv_vals = []
            for costs in results:
                # Update reference point
                reference_point = np.maximum(
                    reference_point,
                    np.max(np.array(list(costs.values())), axis=0))
                # Compute hypervolume
                acc_costs.append(costs)
                pareto = pareto_front(acc_costs)
                pareto = np.array([list(ac.values()) for ac in acc_costs])[pareto]
                pareto = pareto[pareto[:, 0].argsort()]
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)
            budget_list = data["budget_used_total"].values.astype(np.float64)
            seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)
            marker, color = _get_style(
                instance_data[seed]["optimizer"],
                instance_data[seed]["prior_annotations"],
            )
            if pareto_seeds:
                # Plotting Pareto fronts for each seed
                plt.figure(1)
                plt.step(
                    pareto[:, 0],
                    pareto[:, 1],
                    where="post",
                    marker=marker,
                    color=color if not is_single_opt else None,
                    markersize=7,
                    linewidth=1,
                )
                plt.legend([f"{instance}_{seed}" for seed in instance_data])
        plt.figure(1)
        if not pareto_seeds:
            plt.step(
                pareto[:, 0],
                pareto[:, 1],
                where="post",
                marker=marker,
                color=color if not is_single_opt else None,
                markersize=7,
                linewidth=1,
            )
            plt.legend(list(agg_data.keys()))
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.grid(visible=True)
        plt.title(f"Pareto front for\n{plot_title}")
        pareto_save_dir = exp_dir / "plots" / "pareto"
        pareto_save_dir.mkdir(parents=True, exist_ok=True)


        seed_hv_df = pd.DataFrame(seed_hv_dict)
        seed_hv_df = seed_hv_df.ffill(axis=0)
        seed_hv_df = seed_hv_df.dropna(axis=0)
        means = pd.Series(seed_hv_df.mean(axis=1), name=f"means_{instance}")
        std = pd.Series(seed_hv_df.std(axis=1), name=f"std_{instance}")
        means = means.cummax()
        means = means.drop_duplicates()
        std = std.loc[means.index]
        means[budget] = means.iloc[-1]
        std[budget] = std.iloc[-1]

        # Plotting Hypervolumes
        plt.figure(2)
        plt.plot(
            means,
            marker=marker,
            color=color if not is_single_opt else None,
            linestyle="-",
        )
        plt.fill_between(
            means.index,
            means - std,
            means + std,
            alpha=0.2,
            color=color if not is_single_opt else None,
            edgecolor=color if not is_single_opt else None,
            linewidth=2,
        )
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.grid(visible=True)
        plt.legend(list(agg_data.keys()))
        plt.title(f"Hypervolume over iterations plot for\n{plot_title}")
        hv_save_dir = exp_dir / "plots"/ "hypervolume"
        hv_save_dir.mkdir(parents=True, exist_ok=True)


    plot_title = plot_title.replace("\n", "")
    plt.figure(1)
    plt.savefig(pareto_save_dir / f"Pareto_plot_{plot_title.replace(' ', '_')}.png")

    plt.figure(2)
    plt.savefig(hv_save_dir / f"Hypervolume_plot_{plot_title.replace(' ', '_')}.png")
    plt.show()


def agg_data(
    exp_dir: Path,
    *,
    pareto_seeds: bool = False,
)-> None:
    """Function to aggregate data from all runs in the experiment directory."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}

    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    budget: int = 0

    for benchmark in benchmarks_in_dir:
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            _df = pd.read_parquet(file)
            instance = (
                _df["optimizer"][0] + ";" + _df["optimizer_hyperparameters"][0] + ";" +
                _df["prior_annotations"][0]
            )
            seed = int(_df["seed"][0])
            if instance not in agg_dict:
                agg_dict[instance] = {}
            agg_dict[instance][seed] = {
                "results": _df["results"],
                "budget_used_total": _df["budget_used_total"],
                "optimizer": _df["optimizer"][0],
                "prior_annotations": _df["prior_annotations"][0],
                "plot_title": (
                    f"{instance}"
                    f" on \n{benchmark}"
                ),
            }
            budget = _df["budget_used_total"].iloc[-1]
        is_single_opt = False
        if len(agg_dict) == 1:
            is_single_opt = True
        create_plots(
            agg_data=agg_dict,
            exp_dir=exp_dir,
            benchmark=benchmark,
            budget=budget,
            is_single_opt=is_single_opt,
            pareto_seeds=pareto_seeds,
        )


def make_subplots(
    exp_dir: Path,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    num_plots = len(list(exp_dir.rglob("*.png")))
    fig = pplt.figure(tight=True, refwidth="5em")
    axs = fig.subplots(
        nrows = 2 if num_plots > 2 else 1,  # noqa: PLR2004
        ncols = num_plots // 2 if num_plots > 2 else num_plots, # noqa: PLR2004
    )
    for ax, file in zip(axs, exp_dir.rglob("*.png")):  # noqa: B905
        if "pareto" not in file.name.lower():
            continue
        img = plt.imread(file)
        ax.grid(visible=False)
        ax.imshow(img)
        ax.axis("off")
    save_dir = exp_dir / "plots" / "subplots"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "all_pareto_subplots.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", "-e",
        type=str,
        required=True,
        help="Main experiment directory containing the runs to plot."
    )
    parser.add_argument(
        "--make_subplots", "-s",
        action="store_true",
        help="make subplot for all plots in the same experiment directory."
    )
    parser.add_argument(
        "--pareto_seeds", "-p",
        action="store_true",
        help="plot pareto front for each seed."
    )
    args = parser.parse_args()
    exp_dir: Path = DEFAULT_RESULTS_DIR / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}

    if args.make_subplots:
        plots_dir = exp_dir / "plots"
        match plots_dir.exists(), len(list(plots_dir.iterdir())):
            case True, 1:
                logger.info("Only one plot found. Exiting.")
            case _, 0:
                agg_data(exp_dir)
                make_subplots(exp_dir)
            case _:
                make_subplots(exp_dir)
    else:
        agg_data(
            exp_dir=exp_dir,
            pareto_seeds=args.pareto_seeds
        )
        # make_subplots(exp_dir)
