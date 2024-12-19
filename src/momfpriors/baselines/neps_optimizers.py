from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import neps
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpoglue import BenchmarkDescription, Config, Problem, Query, Result

from momfpriors.benchmarks.utils import objective_fn_wrapper
from momfpriors.optimizer import Abstract_NonAskTellOptimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def neps_so_pipeline():
    pass


def neps_mo_scalarized(
    objective_fn: Callable,
    objectives: list[str],
    scalarization_weights: Mapping[str, float],
    **config: Mapping[str, Any]
) -> Mapping[str, Any]:
    results = objective_fn_wrapper(objective_fn, **config)
    logger.info(results)
    scalarized_objective = sum(
        scalarization_weights[obj] * results[obj] for obj in objectives
    )
    return {
        "loss": scalarized_objective,
        "cost": None,
        "info_dict": results
    }

class NepsOptimizer(Abstract_NonAskTellOptimizer):
    name = "NepsOptimizer"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int = 0,
        root_dir: str | Path = "results",
    ) -> None:
        self.problem = problem
        self.space = problem.config_space
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        self.searcher = self.problem.optimizer_hyperparameters.get(
            "searcher", "random_search"
        )
        self.name = f"NepsOptimizer_{self.searcher}"
        self.seed = seed
        self.working_dir = root_dir / "neps_dir" / self.name


    def optimize(
        self,
        scalarization_weights: Mapping[str, float] | None = None,
    ) -> None:
        objectives = self.problem.get_objectives()
        objectives = objectives if isinstance(objectives, list) else [objectives]
        benchmark = self.problem.benchmark.load(self.problem.benchmark)
        if len(objectives) > 1:
            if not scalarization_weights:
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

        np.random.seed(self.seed)  # noqa: NPY002
        neps.run(
            run_pipeline=run_pipeline,
            pipeline_space=self.space,
            root_directory=self.working_dir,
            max_evaluations_total=self.problem.budget.total,
            post_run_summary=True,
            overwrite_working_directory=True,
            searcher=self.searcher,
        )

