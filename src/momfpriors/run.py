from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from hpoglue import FunctionalBenchmark

from momfpriors.baselines import OPTIMIZERS
from momfpriors.benchmarks import BENCHMARKS
from momfpriors.constants import DEFAULT_ROOT_DIR


def run(
    optimizer: tuple[str, Mapping[str, Any]],
    benchmark: tuple[str, str | list[str]],
    seed: int = 0,
    num_iterations: int = 1000,
    prior_path: str | None = None,
    root_dir: str | Path = "results",
    **kwargs: Any
) -> None:

    optimizer_name, optimizer_kwargs = optimizer
    benchmark_name, objectives = benchmark

    if isinstance(objectives, str):
        objectives = [objectives]

    optimizer = OPTIMIZERS[optimizer_name]
    benchmark = BENCHMARKS[benchmark_name]
    if isinstance(benchmark, FunctionalBenchmark):
        benchmark = benchmark.description

    if prior_path is not None:
        with prior_path.open("r") as file:
            prior = yaml.safe_load(file)
    else:
        prior = None

    if optimizer_name == "NepsOptimizer":
        opt = optimizer(
            space = benchmark.config_space,
            seed = seed,
            root_dir = root_dir,
            **optimizer_kwargs
        )
        opt(
            benchmark = benchmark,
            objectives = objectives,
            max_evaluations = num_iterations,
            scalarization_weights = kwargs.get("scalarization_weights"),
        )

if __name__ == "__main__":
    run(
        optimizer = (
            "NepsOptimizer",
            {
                "searcher": "bayesian_optimization",
            }
        ),
        benchmark = (
            "MOMFPark",
            ["value1", "value2"]
        ),
        seed=1,
        num_iterations=100,
        prior_path=None,
        root_dir=DEFAULT_ROOT_DIR,
    )

