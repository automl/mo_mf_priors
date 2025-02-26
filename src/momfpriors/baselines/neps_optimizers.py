from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from hpoglue import Config, Problem, Query, Result
from hpoglue.env import Env
from neps import AskAndTell, algorithms
from neps.space.parsing import convert_configspace

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NepsOptimizer(Abstract_AskTellOptimizer):
    """Base class for Neps Optimizers."""
    name = "NepsOptimizer"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="NEPS-0.12.3",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.3",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int = 0,
        working_directory: str | Path = DEFAULT_RESULTS_DIR,
        **kwargs: Any,
    ) -> None:
        self.problem = problem
        self.space = convert_configspace(self.problem.config_space)
        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.searcher = self.problem.optimizer_hyperparameters.get(
            "searcher", "bayesian_optimization"
        )
        kwargs.pop("searcher", None)
        self.seed = seed
        self.working_dir = working_directory / "neps_dir" / self.name

        np.random.seed(self.seed)  # noqa: NPY002
        opt = algorithms.PredefinedOptimizers[self.searcher](
            space = self.space,
            **kwargs
        )
        self.optimizer = AskAndTell(opt)
        self.trial_counter = 0


    def ask(self) -> Query:
        trial = self.optimizer.ask() # TODO: Figure out fidelity
        return Query(
            config = Config(config_id=self.trial_counter, values=trial.config),
            fidelity=None,
            optimizer_info=trial
        )

    def tell(self, result: Result) -> None:
        costs = list(result.values.values())
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=costs
        )


class NepsRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Bayesian Optimization in Neps."""

    name = "NepsRW"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.12.2",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.2",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int = 0,
        working_directory: str | Path = DEFAULT_RESULTS_DIR,
        scalarization_weights: Mapping[str, float] | None = None,
        searcher: str = "bayesian_optimization",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            problem=problem,
            seed=seed,
            working_directory=working_directory,
            searcher=searcher,
        )
        self.objectives = self.problem.get_objectives()
        self.scalarization_weights = scalarization_weights
        if not self.scalarization_weights:
            self.scalarization_weights = {
                obj: 1.0/len(self.objectives) for obj in self.objectives
            }


    def tell(self, result: Result) -> None:
        costs = {obj: result.values[obj] for obj in self.objectives}
        scalarized_objective = sum(
            self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
        )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=scalarized_objective
        )