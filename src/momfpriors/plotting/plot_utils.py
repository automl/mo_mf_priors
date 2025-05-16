from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from momfpriors.plotting.plot_styles import (
    COLORS,
    MARKERS,
    OPTIMIZER_LABELS1,
    OPTIMIZER_LABELS2,
    PREFERENCE_LABELS1,
    PREFERENCE_LABELS2,
)


def dynamic_reference_point(
    loss_vals: pd.Series | list[Mapping[str, float]]
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if isinstance(loss_vals, pd.Series):
        loss_vals = loss_vals.to_numpy()
    loss_vals = np.array(
        [
            list(cost.values())
            for cost in loss_vals
        ]
    )
    worst_point = np.max(loss_vals, axis=0)
    reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
    reference_point[reference_point == 0] = 1e-12
    return reference_point


def pareto_front(
    costs: pd.Series | list[Mapping[str, float]] | np.array,
) -> np.array:
    """Function to calculate the pareto front from a pandas Series
    of Results, i.e., Mapping[str, float] objects.
    """
    if not isinstance(costs, np.ndarray):
        costs: np.array = np.array([list(cost.values()) for cost in costs])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto


def get_style(instance: str) -> tuple[str, str, str, str | None]:
    """Function to get the plotting style for a given instance."""
    prior_annot = instance.split("priors=")[-1] if "priors=" in instance else None
    opt = instance.split(";")[0]
    color = COLORS.get(opt)
    if prior_annot:
        color = COLORS.get(f"{opt}")
    marker = MARKERS.get(prior_annot, "s")
    if color is None:
        print(f"No color found for {opt}")
    return marker, color, opt, prior_annot


def edit_legend_labels(labels: list[str]) -> list[str]:
    """Edit the legend labels to be more readable."""
    new_labels = []
    for label in labels:
        prior_annot = ""
        _label = label
        if "_w_continuations" in _label:
            _label = _label.replace("_w_continuations", "")
        if ";priors=" in _label:
            _label, prior_annot = _label.split(";priors=")
            prior_annot = f" ({prior_annot})"
        _label = PREFERENCE_LABELS2.get(
            _label, OPTIMIZER_LABELS2.get(_label, _label)
        )
        _label = f"{_label}{prior_annot}"
        new_labels.append(_label)
    return new_labels


def avg_seed_dfs_for_ranking(
    seed_means_dict: dict[str | int, dict[str, pd.Series]],
) -> dict[int, dict[str, pd.Series]]:
    """Function to average prior seeds and move them into their
    respective seed dicts.
    """
    final_dict = {}
    seed_dfs = {}
    for seed, instances in seed_means_dict.items():
        if isinstance(seed, str) and "_" in seed:
            assert len(instances) == 1
            _seed = int(seed.split("_")[0])
            if _seed not in seed_dfs:
                seed_dfs[_seed] = {}
            instance = next(iter(instances))
            if instance not in seed_dfs[_seed]:
                seed_dfs[_seed][instance] = []
            seed_dfs[_seed][instance].append(instances[instance])
        elif seed not in final_dict:
            final_dict[seed] = instances
    for seed, instances in seed_dfs.items():
        for instance, df_list in instances.items():
            if seed not in final_dict:
                final_dict[seed] = {}
            else:
                assert instance not in final_dict[seed]
            final_dict[seed][instance] = pd.DataFrame(df_list).mean()
    return final_dict

reference_points_dict = {

    # MOMFPark
    "MOMFPark": {"value1": 1, "value2": 1},

    # MOMFBraninCurrin
    "MOMFBraninCurrin": {"value1": 1, "value2": 1},

    # PD1
    "pd1-cifar100-wide_resnet-2048": {"valid_error_rate": 1, "train_cost": 30},
    "pd1-imagenet-resnet-512": {"valid_error_rate": 1, "train_cost": 5000},
    "pd1-lm1b-transformer-2048": {"valid_error_rate": 1, "train_cost": 1000},
    "pd1-translate_wmt-xformer_translate-64": {"valid_error_rate": 1, "train_cost": 20000},

    # JAHSBench
    "jahs-CIFAR10": {"valid_acc": 0, "runtime": 200000},
    "jahs-ColorectalHistology": {"valid_acc": 0, "runtime": 200000},
    "jahs-FashionMNIST": {"valid_acc": 0, "runtime": 200000},

    # YAHPO-LCBench
    "yahpo-lcbench-126026": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 150
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
        "time": 5000
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
        "time": 150
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
        "time": 200
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

fid_perc_momfbo = {
    "MOMFPark": 0.8,
    "pd1-cifar100-wide_resnet-2048": 0.8,
    "pd1-imagenet-resnet-512": 0.8,
    "pd1-lm1b-transformer-2048": 0.8,
    "pd1-translate_wmt-xformer_translate-64": 0.8,
    "yahpo-lcbench-126026": 0.8,
    "yahpo-lcbench-146212": 0.8,
    "yahpo-lcbench-168330": 0.8,
    "yahpo-lcbench-168868": 0.8,
    "jahs-CIFAR10": 0.9,
    "jahs-ColorectalHistology": 0.9,
    "jahs-FashionMNIST": 0.9,
    "yahpo-lcbench-167190": 0.9,
    "yahpo-lcbench-168910": 0.9,
    "yahpo-lcbench-189906": 0.9,
    "yahpo-lcbench-3945": 0.9,
    "yahpo-lcbench-7593": 0.9,
    "yahpo-lcbench-34539": 0.9,
    "yahpo-lcbench-126025": 0.9,
    "yahpo-lcbench-126029": 0.9,
    "yahpo-lcbench-167104": 0.9,
    "yahpo-lcbench-167149": 0.9,
    "yahpo-lcbench-167152": 0.9,
    "yahpo-lcbench-167161": 0.9,
    "yahpo-lcbench-167168": 0.9,
    "yahpo-lcbench-167181": 0.9,
    "yahpo-lcbench-167184": 0.9,
    "yahpo-lcbench-167185": 0.9,
    "yahpo-lcbench-167200": 0.9,
    "yahpo-lcbench-167201": 0.9,
    "yahpo-lcbench-168329": 0.9,
    "yahpo-lcbench-168331": 0.9,
    "yahpo-lcbench-168335": 0.9,
    "yahpo-lcbench-168908": 0.9,
    "yahpo-lcbench-189354": 0.9,
    "yahpo-lcbench-189862": 0.9,
    "yahpo-lcbench-189865": 0.9,
    "yahpo-lcbench-189866": 0.9,
    "yahpo-lcbench-189873": 0.9,
    "yahpo-lcbench-189905": 0.9,
    "yahpo-lcbench-189908": 0.9,
    "yahpo-lcbench-189909": 0.9
}