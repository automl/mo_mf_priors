from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import neps
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpoglue import BenchmarkDescription, Config, Query, Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def neps_so_pipeline():
    pass


def neps_objective_fn_wrapper(
    objective_fn: Callable,
    **config: Mapping[str, Any]
) -> Mapping[str, Any]:
    query = Query(
        config=Config(config_id=None, values=config),
        fidelity=None,
    )
    result: Result = objective_fn(query)
    return result.values


def neps_mo_scalarized(
    objective_fn: Callable,
    objectives: list[str],
    scalarization_weights: Mapping[str, float],
    **config: Mapping[str, Any]
) -> Mapping[str, Any]:
    results = neps_objective_fn_wrapper(objective_fn, **config)
    scalarized_objective = sum(
        scalarization_weights[obj] * results[obj] for obj in objectives
    )
    return {
        "loss": scalarized_objective,
        "cost": None,
        "info_dict": results
    }

class NepsOptimizer:
    name = "NepsOptimizer"

    def __init__(
        self,
        space: ConfigurationSpace,
        seed: int = 0,
        root_dir: str | Path = "results",
        **kwargs
    ):
        self.space = space
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        self.working_dir = root_dir / "neps_dir"
        self.searcher = kwargs.get("searcher", "random_search")
        self.name = f"NepsOptimizer_{self.searcher}"
        self.seed = seed


    def __call__(
        self,
        benchmark: BenchmarkDescription,
        objectives: str | list[str],
        max_evaluations: int = 1000,
        scalarization_weights: Mapping[str, float] | None = None,
    ) -> None:
        objectives = objectives if isinstance(objectives, list) else [objectives]
        benchmark = benchmark.load(benchmark)
        if len(objectives) > 1:
            if scalarization_weights is None:
                scalarization_weights = {
                    f"{obj}": 1.0/len(objectives) for obj in objectives
                }
            run_pipeline = partial(
                neps_mo_scalarized,
                objective_fn=benchmark.query,
                objectives=objectives,
                scalarization_weights=scalarization_weights)
        else:
            run_pipeline = neps_so_pipeline

        np.random.seed(self.seed)
        neps.run(
            run_pipeline=run_pipeline,
            pipeline_space=self.space,
            root_directory=self.working_dir,
            max_evaluations_total=max_evaluations,
            post_run_summary=True,
            overwrite_working_directory=True,
            searcher=self.searcher,
        )

