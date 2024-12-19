from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from hpoglue import FunctionalBenchmark, Problem

from momfpriors.baselines import OPTIMIZERS
from momfpriors.benchmarks import BENCHMARKS
from momfpriors.constants import DEFAULT_ROOT_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer, Abstract_NonAskTellOptimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)


def _run(
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

    problem = Problem.problem(
        optimizer = optimizer,
        optimizer_hyperparameters=optimizer_kwargs,
        benchmark=benchmark,
        objectives=objectives,
        budget=num_iterations,
        multi_objective_generation="mix_metric_cost",
    )

    benchmark = benchmark.load(benchmark)

    logger.info(f"Running {optimizer_name} on {benchmark_name} with objectives {objectives}")

    if issubclass(optimizer, Abstract_NonAskTellOptimizer):
        opt = optimizer(
            problem=problem,
            seed=seed,
            root_dir=root_dir,
        )
        opt.optimize(kwargs)

    elif issubclass(optimizer, Abstract_AskTellOptimizer):
        opt = optimizer(
            problem=problem,
            seed=seed,
            working_directory=root_dir/"Optimizers_cache",
        )
        for i in range(num_iterations):
            query = opt.ask()
            result = benchmark.query(query)
            opt.tell(result)
            logger.info(f"Iteration {i+1}/{num_iterations}: {result.values}")

    else:
        raise TypeError(
            f"Unknown optimizer type {optimizer}"
            f"Expected {Abstract_NonAskTellOptimizer} or"
            f"{Abstract_AskTellOptimizer}"
        )


if __name__ == "__main__":
    _run(
        # optimizer = (
        #     "RandomSearch",
        #     {}
        # ),
        optimizer = (
            "NepsOptimizer",
            {
                "searcher": "bayesian_optimization",
            }
        ),
        # optimizer = (
        #     "SMAC_ParEGO",
        #     {}
        # ),
        benchmark = (
            "MOMFPark",
            ["value1", "value2"]
        ),
        # benchmark = (
        #     "hpobench_xgb_9977",
        #     ["function_value", "cost"]
        # ),
        seed=1,
        num_iterations=10,
        prior_path=None,
        root_dir=DEFAULT_ROOT_DIR,
    )

