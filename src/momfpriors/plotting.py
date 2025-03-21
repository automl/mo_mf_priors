from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import Hypervolume

from momfpriors.constants import DEFAULT_RESULTS_DIR, DEFAULT_ROOT_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        "time": 200000
    },
    "yahpo-lcbench-167190": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200000
    },
    "yahpo-lcbench-168330": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200000
    },
    "yahpo-lcbench-168910": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200000
    },
    "yahpo-lcbench-189906": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200000
    },
}


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


def _get_style(instance: str) -> tuple[str, str]:
    """Function to get the plotting style for a given instance."""
    prior_annot = instance.split("priors=")[-1] if "priors=" in instance else None
    opt = instance.split(";")[0]
    color = style_dict["colors"].get(opt)
    marker = style_dict["markers"].get(prior_annot, "s")
    return marker, color


def plot_average_rank(
    ranks: dict,
    budget: int,
    exp_dir: Path,
    *,
    no_save: bool = False,
    benchmark: str | None = None,
) -> None:
    """Plots the average rank of optimizers over iterations with standard error."""
    def _mean(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(_dfs).reset_index().groupby("index").mean()

    def _sem(_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(_dfs).reset_index().groupby("index").sem()

    # Get mean across all benchmarks, for each seed
    ranks_per_seed_averaged_over_benchmarks = {
        seed: _mean(ranks[seed].values()) for seed in ranks
    }

    # Average over all seeds
    mean_ranks = _mean(ranks_per_seed_averaged_over_benchmarks.values())
    sem_ranks = _sem(ranks_per_seed_averaged_over_benchmarks.values())

    # Sort by the last iteration
    mean_ranks = mean_ranks.sort_values(by=budget, axis=1, ascending=False)
    sem_ranks = sem_ranks[mean_ranks.columns]

    # Plot settings
    plt.figure(figsize=(12, 6))

    # Plotting
    for instance in mean_ranks.columns:
        means = mean_ranks[instance]
        sems = sem_ranks[instance]

        marker, color = _get_style(instance)
        plt.plot(
            means,
            linestyle="-",
            label=instance,
            marker=marker,
            color=color,
        )
        plt.fill_between(
            means.index,
            means - sems,
            means + sems,
            alpha=0.2,
            linewidth=2,
            color=color,
            edgecolor=color,
        )

    # Labels and legend
    plot_title = "Average_Rank_of_Optimizers_over_Iterations_across_all_Benchmarks_and_seeds"
    if benchmark:
        plot_title = plot_title.replace(
            "across_all_Benchmarks_and_seeds",
            f"for_{benchmark}_and_all_seeds"
        )
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Rank (Lower is Better)")
    plt.title(plot_title)
    plt.xticks(np.arange(5, budget + 1, step=5))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.05),
        fontsize=12
    )
    plt.tight_layout()
    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.6)

    # Save the plot
    if not no_save:
        plt.savefig(exp_dir / "plots" / f"{plot_title}.png")
    plt.show()





def create_plots(  # noqa: C901, PLR0912, PLR0915
    agg_data: Mapping[str, Mapping[str, Any]],
    exp_dir: Path,
    benchmark: str,
    objectives: list[str],
    budget: int,
    seed_for_pareto: int,
    plot_opt: str | None = None,
    *,
    is_single_opt: bool = False,
    no_save: bool = False,
) -> None:
    """Function to plot the dominated hypervolume over
    iterations and pareto fronts from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    print("================================================")
    logger.info(f"Plots for runs on benchmark: {benchmark}")
    plt.figure(1, figsize=(10, 10))
    plt.figure(2, figsize=(10, 10))

    seed_means_dict = {}
    means_dict = {}

    reference_point = np.array([
        reference_points_dict[benchmark][obj]
        for obj in objectives
    ])
    if plot_opt:
        plt.figure(3, figsize=(10, 10))
    for instance, instance_data in agg_data.items():
        logger.info(f"Plots for instance: {instance}")
        seed_hv_dict = {}
        for seed, data in instance_data.items():
            results = data["results"]
            plot_title = data["plot_title"]
            keys = list(results[0].keys())
            acc_costs = []
            pareto = None
            hv_vals = []
            for i, costs in enumerate(results, start=1):
                # Compute hypervolume
                # if "jahs" in benchmark:
                #     costs["valid_acc"] = -costs["valid_acc"]
                acc_costs.append(costs)
                pareto = pareto_front(acc_costs)
                pareto = np.array([list(ac.values()) for ac in acc_costs])[pareto]
                pareto = pareto[pareto[:, 0].argsort()]
                hv = Hypervolume(ref_point=reference_point)
                hypervolume = hv.do(pareto)
                hv_vals.append(hypervolume)

                if i == budget:
                    break

            budget_list = data["budget_used_total"].values.astype(np.float64)[:budget]
            seed_hv_dict[seed] = pd.Series(hv_vals, index=budget_list)
            marker, color = _get_style(instance)

            # Accumulating incumbents for every optimizer instance per seed per benchmark
            seed_incumbents = seed_hv_dict[seed].cummax()
            if seed not in seed_means_dict:
                seed_means_dict[seed] = {}
            seed_means_dict[seed][instance] = seed_incumbents

            match plot_opt:
                case None:
                    pass
                case "all":
                # Plotting Pareto fronts for each seed
                    plt.figure(3)
                    plt.step(
                        pareto[:, 0],
                        pareto[:, 1],
                        where="post",
                        marker=marker,
                        label=f"{instance}_{seed}",
                        markersize=7,
                        linewidth=1,
                    )
                    plt.legend([f"{instance}_{seed}" for seed in instance_data])
                    plt.xlabel(keys[0])
                    plt.ylabel(keys[1])
                    plt.grid(visible=True)
                    plt.title(f"Multiple seeds pareto plot for \n{instance}")
                case str():
                    if plot_opt in instance:
                        plt.figure(3)
                        plt.step(
                            pareto[:, 0],
                            pareto[:, 1],
                            where="post",
                            marker=marker,
                            label=f"{instance}_{seed}",
                            markersize=7,
                            linewidth=1,
                        )
                        plt.legend()
                        plt.xlabel(keys[0])
                        plt.ylabel(keys[1])
                        plt.grid(visible=True)
                        plt.title(f"Pareto front for \n{plot_title}")
                case _:
                    raise ValueError(f"Invalid plot_opt: {plot_opt}.")
            if seed == seed_for_pareto:
                plt.figure(1)
                plt.step(
                    pareto[:, 0],
                    pareto[:, 1],
                    where="post",
                    marker=marker,
                    color=color if not is_single_opt else None,
                    label=f"{instance}_{seed}",
                    markersize=7,
                    linewidth=1,
                )
        if plot_opt and (plot_opt == "all" or plot_opt in instance):
            pareto_save_dir = exp_dir / "plots" / str(budget) /"pareto"
            pareto_save_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(3)
            if not no_save:
                plt.savefig(pareto_save_dir / f"Multiple seeds pareto plot for {instance}.png")
            plt.clf()

        if len(agg_data) > 1:
            plot_title = f"Optimizers on \n{benchmark} for {budget} iterations"

        plt.figure(1)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            fontsize=12
        )
        plt.tight_layout()
        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.grid(visible=True)
        plt.title(f"Pareto front for\n{plot_title}")
        pareto_save_dir = exp_dir / "plots" / str(budget) / "pareto"
        pareto_save_dir.mkdir(parents=True, exist_ok=True)

        # Aggregating Hypervolumes - calculating means, cumulative max and std_error
        seed_hv_df = pd.DataFrame(seed_hv_dict)
        means = pd.Series(seed_hv_df.mean(axis=1), name=f"means_{instance}")
        ste = pd.Series(seed_hv_df.sem(axis=1), name=f"std_{instance}")
        means = means.cummax()
        means = means.drop_duplicates()
        ste = ste.loc[means.index]
        means[budget] = means.iloc[-1]
        ste[budget] = ste.iloc[-1]

        # Plotting Hypervolumes
        plt.figure(2)
        plt.plot(
            means,
            marker=marker,
            color=color if not is_single_opt else None,
            linestyle="-",
            label=instance,
        )
        plt.fill_between(
            means.index,
            means - ste,
            means + ste,
            alpha=0.2,
            color=color if not is_single_opt else None,
            edgecolor=color if not is_single_opt else None,
            linewidth=2,
        )
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            fontsize=12
        )
        plt.tight_layout()
        plt.xlabel("Iteration")
        plt.xticks(np.arange(1, budget + 1, 1))     # TrialBudget
        plt.ylabel("Hypervolume")
        plt.grid(visible=True)
        plt.title(f"Hypervolume over iterations plot for\n{plot_title}")
        hv_save_dir = exp_dir / "plots"/ str(budget) / "hypervolume"
        hv_save_dir.mkdir(parents=True, exist_ok=True)


    plot_title = plot_title.replace("\n", "")
    if not no_save:
        plt.figure(1)
        plt.savefig(pareto_save_dir / f"Pareto_plot_{plot_title.replace(' ', '_')}.png")

        plt.figure(2)
        plt.savefig(hv_save_dir / f"Hypervolume_plot_{plot_title.replace(' ', '_')}.png")
    plt.show()

    plt.close("all")

    # Ranking every optimizer instance per seed per benchmark
    for seed, instances in seed_means_dict.items():
        _df = pd.DataFrame(instances)
        _df = _df.rank(axis=1, method="average", ascending=False)
        means_dict[seed] = _df
    return means_dict


def agg_data(  # noqa: C901
    exp_dir: Path,
    plot_opt: str | None = None,
    plot_iters: list[int] | None = None,
    *,
    no_save: bool = False,
)-> None:
    """Function to aggregate data from all runs in the experiment directory."""
    agg_dict: Mapping[str, Mapping[str, Any]] = {}
    means_dict = {}

    benchmarks_in_dir = [
        (f.name.split("benchmark=")[-1].split(".")[0])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    benchmarks_in_dir.sort()
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")

    budget: int = 0


    with (exp_dir / "exp.yaml").open("r") as f:
        exp_config = yaml.safe_load(f)

    seed_for_pareto = exp_config.get("seeds")[0]
    iterations = exp_config.get("budget")

    print(type(plot_iters))

    if plot_iters and isinstance(plot_iters, int):
        plot_iters = [plot_iters]
    plot_iters = plot_iters if plot_iters else []

    if iterations not in plot_iters:
        assert any(plot_iters) < iterations, (
            "Iterations to plot should be less than the total budget."
        )
        plot_iters.append(iterations)


    for iteration in plot_iters:
        logger.info(f"Plotting for iteration: {iteration}")

        for benchmark in benchmarks_in_dir:
            objectives = []
            for file in exp_dir.rglob("*.parquet"):
                if benchmark not in file.name:
                    continue
                _df = pd.read_parquet(file)

                objectives = _df["objectives"][0]

                assert len(objectives) == 2, ( # noqa: PLR2004
                    "More than 2 objectives found in results file: "
                    f"{objectives}. "
                    "Can only plot pareto front for 2D cost space."
                )

                minimize: dict[str, bool] = _df["minimize"][0]

                _results = _df["results"].apply(
                    lambda x, objectives=objectives, minimize=minimize: {
                        k: x[k] if minimize[k] else -x[k] for k in objectives
                    }
                )

                if _df["prior_annotations"][0] is not None:
                    annotations = "-".join(
                        [a.split("=")[-1] for a in _df["prior_annotations"][0].split(",")]
                    )

                instance = (
                    _df["optimizer"][0] +
                    (
                        ";" + _df["optimizer_hyperparameters"][0]
                        if "default" not in _df["optimizer_hyperparameters"][0]
                        else ""
                    ) +
                    (f";priors={annotations}" if annotations else "")
                )
                seed = int(_df["seed"][0])
                if instance not in agg_dict:
                    agg_dict[instance] = {}
                agg_dict[instance][seed] = {
                    "results": _results,
                    "budget_used_total": _df["budget_used_total"],
                    "plot_title": (
                        f"{instance}"
                        f" on \n{benchmark}"
                    ),
                }
                budget = _df["budget_used_total"].iloc[-1]
            is_single_opt = False
            if len(agg_dict) == 1:
                is_single_opt = True
            assert len(objectives) > 0, "Objectives not found in results file."
            seed_dict_per_bench = create_plots(
                agg_data=agg_dict,
                exp_dir=exp_dir,
                benchmark=benchmark,
                budget=iteration,
                is_single_opt=is_single_opt,
                plot_opt=plot_opt,
                seed_for_pareto=seed_for_pareto,
                objectives=objectives,
                no_save=no_save,
            )
            agg_dict = {}
            for _seed, rank_df in seed_dict_per_bench.items():
                if _seed not in means_dict:
                    means_dict[_seed] = {}
                means_dict[_seed][benchmark] = rank_df
            # Per benchmark ranking plots
            plot_average_rank(
                means_dict,
                iteration,
                exp_dir,
                no_save=no_save,
                benchmark=benchmark,
            )
    # Plotting average ranks over all benchmarks and seeds
    plot_average_rank(means_dict, budget, exp_dir, no_save=no_save)


def make_subplots(
    exp_dir: Path,
    iteration: int | None = None,
    *,
    no_save: bool = False,
) -> None:
    """Function to make subplots for all plots in the same experiment directory."""
    if isinstance(iteration, list):
        iteration = iteration[0]
    pareto_plots_dir = exp_dir / "plots" / str(iteration) / "pareto"
    hv_plots_dir = exp_dir / "plots" / str(iteration) / "hypervolume"
    str_to_match = "*iterations.png"
    num_plots = len(list(pareto_plots_dir.rglob(str_to_match)))
    print(f"Found {num_plots} plots.")
    # assert num_plots == len(list(hv_plots_dir.rglob(str_to_match))), "Number of plots do not match."


    def plot_subplots(dir: Path, type: Literal["pareto", "hypervolume"]) -> None:
        image_paths = sorted(dir.rglob(str_to_match))
        images = [mpimg.imread(img) for img in image_paths]
        # nrows = 2 if num_plots > 2 else 1 # noqa: PLR2004
        # ncols = num_plots // 2 if num_plots > 2 else num_plots # noqa: PLR2004
        nrows, ncols = 2, 2
        while num_plots > nrows * ncols:
            ncols += max(1, (num_plots - nrows * ncols) // nrows)
            if ncols > 4:   # noqa: PLR2004
                nrows += 1
                ncols = 4
        if ncols <1 or nrows < 1:
            logger.error("Just one plot found. Exiting.")
            return
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            figsize=(20, 10),
        )
        axs = axs.flatten() if num_plots > 1 else [axs]
        # print(len(axs))
        for i, img in enumerate(images):
            # print(i)
            axs[i].imshow(img)
            axs[i].axis("off")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.suptitle(f"All {type} plots_{iteration=}")
        save_dir = dir.parent / "subplots"
        save_dir.mkdir(parents=True, exist_ok=True)
        if not no_save:
            plt.savefig(save_dir / f"all_{type}_subplots_{iteration=}.png", dpi = 150)
        plt.show()

    plot_subplots(pareto_plots_dir, "pareto")
    plot_subplots(hv_plots_dir, "hypervolume")


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
        "--plot_opt", "-o",
        type=str,
        default=None,
        help="Plot all pareto fronts only for the given optimizer."
    )
    parser.add_argument(
        "--plot_for_iterations", "-i",
        type=int,
        nargs="+",
        default=None,
        help="Plot pareto fronts and Hypervolumes for the given iterations."
    )
    parser.add_argument(
        "--no_save", "-ns",
        action="store_true",
        help="Do not save the plots."
    )
    args = parser.parse_args()
    exp_dir: Path = DEFAULT_RESULTS_DIR / args.exp_dir

    agg_dict: Mapping[str, Mapping[str, Any]] = {}

    if args.make_subplots:
        assert args.plot_for_iterations, "Provide iterations to make subplots."
        plots_dir = exp_dir / "plots" / str(args.plot_for_iterations[0])
        # print(len(list((plots_dir/"hypervolume").rglob("*.png"))))
        match (
            plots_dir.exists(),
            len(list((plots_dir/"hypervolume").rglob("*.png")))
        ):
            case True, 1:
                logger.info("Only one plot found. Exiting.")
            case _, 0:
                agg_data(
                    exp_dir=exp_dir,
                    plot_opt=args.plot_opt,
                    plot_iters=args.plot_for_iterations,
                    no_save=args.no_save
                )
                make_subplots(exp_dir, args.plot_for_iterations, no_save=args.no_save)
            case _:
                make_subplots(exp_dir, args.plot_for_iterations, no_save=args.no_save)
    else:
        agg_data(
            exp_dir=exp_dir,
            plot_opt=args.plot_opt,
            plot_iters=args.plot_for_iterations,
            no_save=args.no_save
        )
