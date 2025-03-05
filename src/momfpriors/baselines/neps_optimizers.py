from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import neps
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
        space: neps.SearchSpace,
        optimizer: AskAndTell,
        seed: int = 0,
        working_directory: str | Path = DEFAULT_RESULTS_DIR,
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        self.problem = problem
        self.space = space

        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.seed = seed
        self.working_dir = working_directory

        np.random.seed(self.seed)  # noqa: NPY002
        self.optimizer = optimizer
        self.trial_counter = 0


    def ask(self) -> Query:
        """Ask the optimizer for a new trial."""
        trial = self.optimizer.ask() # TODO: Figure out fidelity
        fidelity = None
        match self.problem.fidelities:
            case None:
                pass
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsOptimizer.")
            case (fid_name, fidelity):
                _fid_value = trial.config.pop(fid_name)
                fidelity = (fid_name, _fid_value)
            case _:
                raise TypeError(
                    "Fidelity must be a tuple or a Mapping. \n"
                    f"Got {type(self.problem.fidelities)}."
                )
        self.trial_counter += 1
        return Query(
            config = Config(config_id=self.trial_counter, values=trial.config),
            fidelity=fidelity,
            optimizer_info=trial
        )

    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        # single objective tell(), must be overridden by multi-objective optimizers
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
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        self.searcher = searcher
        space = convert_configspace(problem.config_space)

        opt = algorithms.PredefinedOptimizers[self.searcher](
            space = space
        )
        optimizer = AskAndTell(opt)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
        )

        self.objectives = self.problem.get_objectives()
        self.scalarization_weights = scalarization_weights
        if not self.scalarization_weights:
            self.scalarization_weights = {
                obj: 1.0/len(self.objectives) for obj in self.objectives
            }


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {obj: result.values[obj] for obj in self.objectives}
        scalarized_objective = sum(
            self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
        )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=scalarized_objective
        )


class NepsHyperbandRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Hyperband for budget allocation in Neps."""

    name = "NepsHyperbandRW"

    support = Problem.Support(
        fidelities=("single",),
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
        searcher: str = "hyperband",
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        self.searcher = searcher
        space = convert_configspace(problem.config_space)

        _fid = None
        min_fidelity: int | float
        max_fidelity: int | float
        match problem.fidelities:
            case None:
                raise ValueError("NepsHyperbandRW requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsHyperbandRW.")
            case (fid_name, fidelity):
                _fid = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
                match _fid.kind:
                    case _ if _fid.kind is int:
                        import neps
                        space.fidelities = {
                            f"{fid_name}": neps.Integer(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _ if _fid.kind is float:
                        import neps
                        space.fidelities = {
                            f"{fid_name}": neps.Float(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid fidelity type: {type(_fid.kind).__name__}. "
                            "Expected int or float."
                        )
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")


        opt = algorithms.PredefinedOptimizers[self.searcher](
            space = space
        )
        optimizer = AskAndTell(opt)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
        )

        self.objectives = self.problem.get_objectives()
        self.scalarization_weights = scalarization_weights
        if not self.scalarization_weights:
            self.scalarization_weights = {
                obj: 1.0/len(self.objectives) for obj in self.objectives
            }

    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {obj: result.values[obj] for obj in self.objectives}
        scalarized_objective = sum(
            self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
        )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=scalarized_objective
        )