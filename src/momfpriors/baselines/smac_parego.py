from __future__ import annotations

from collections.abc import Hashable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from hpoglue import Config, Problem, Query
from hpoglue.env import Env

from momfpriors.baselines.initial_design import RandomInitialDesign
from momfpriors.optimizer import Abstract_AskTellOptimizer

if TYPE_CHECKING:
    from hpoglue import Result
    from smac.runhistory import TrialInfo


def _dummy_target_function(*args: Any, budget: int | float, seed: int) -> NoReturn:
    raise RuntimeError("This should never be called!")


class SMAC_ParEGO(Abstract_AskTellOptimizer):
    """SMAC optimizer using the ParEGO multi-objective algorithm."""

    name = "SMAC_ParEGO"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    env = Env(
        name="SMAC-2.1",
        python_version="3.10",
        requirements=("smac==2.1",)
    )

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        initial_design_size: int | Literal["ndim"] = "ndim",
        consistent_initial_design: bool = True,
        **kwargs: Any # noqa: ARG002
    ):
        """Create a SMAC Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            initial_design_size: Number of shared random startup configurations.
                If "ndim", use the number of searchable hyperparameters. Used
                only when consistent_initial_design is True.
            consistent_initial_design: Use the shared Random Search prefix.
                If False, use the original SMAC facade's default Sobol design.
        """
        initial_design = (
            RandomInitialDesign(
                problem.config_space,
                seed=seed,
                size=initial_design_size,
                budget=problem.budget.total,
            )
            if consistent_initial_design else None
        )

        working_directory.mkdir(parents=True, exist_ok=True)

        self.problem = problem
        self.working_directory = working_directory
        self.config_space = (
            initial_design.config_space if initial_design is not None else problem.config_space
        )
        self._consistent_initial_design = consistent_initial_design
        self._initial_configs = initial_design.configurations if initial_design is not None else []
        self._initial_index = 0
        self._initial_pending: set[int] = set()
        self._fallback_space = deepcopy(self.config_space)
        self._trial_lookup: dict[Hashable, TrialInfo] = {}
        self._seed = seed

        from smac import (  # noqa: PLC0415
            HyperparameterOptimizationFacade as HPOFacade,
            Scenario,
        )
        from smac.multi_objective.parego import ParEGO  # noqa: PLC0415

        scenario = Scenario(
            configspace=self.config_space,
            deterministic=True,
            objectives=problem.get_objectives(),
            n_trials=problem.budget.total,
            use_default_config=False,
            seed=seed,
            output_directory=self.working_directory / "smac-output",
        )
        facade_options: dict[str, Any] = {}
        if initial_design is not None:
            facade_options["initial_design"] = HPOFacade.get_initial_design(
                scenario,
                n_configs=0,
                additional_configs=initial_design.configurations,
            )
        self.optimizer = HPOFacade(
            scenario=scenario,
            logging_level=False,
            target_function=_dummy_target_function,
            multi_objective_algorithm=ParEGO(scenario=scenario),
            intensifier=HPOFacade.get_intensifier(scenario),
            overwrite=True,
            **facade_options,
        )

    def ask(self) -> Query:
        """Ask SMAC for a new config to evaluate."""
        from smac.runhistory import TrialInfo  # noqa: PLC0415

        if not self._consistent_initial_design:
            smac_info = self.optimizer.ask()
        elif self._initial_index < len(self._initial_configs):
            # Emit the full random sequence here: SMAC's initial-design selector
            # deduplicates configurations, whereas Random Search samples with replacement.
            smac_info = TrialInfo(
                config=self._initial_configs[self._initial_index], seed=self._seed,
            )
            self._initial_index += 1
            self._initial_pending.add(id(smac_info))
            self.optimizer.runhistory.add_running_trial(smac_info)
        else:
            if self._initial_pending:
                raise ValueError(
                    "Report all shared initial configurations before requesting a BO trial."
                )
            if len(self.optimizer.runhistory.get_configs()) >= self.config_space.estimate_size():
                # Native SMAC stops after exhausting a finite space. Keep the
                # outer evaluation budget usable by sampling with replacement.
                smac_info = TrialInfo(
                    config=self._fallback_space.sample_configuration(), seed=self._seed,
                )
                self.optimizer.runhistory.add_running_trial(smac_info)
            else:
                # ConfigSelector reads the runhistory on its first iteration and
                # skips the supplied initial configurations already evaluated above.
                smac_info = self.optimizer.ask()
        assert smac_info.instance is None, "We don't do instance benchmarks!"

        config = smac_info.config
        raw_config = dict(config)
        config_id = str(self.optimizer.intensifier.runhistory.config_ids[config])

        return Query(
            config=Config(config_id=config_id, values=raw_config),
            fidelity=None,
            optimizer_info=smac_info,
        )

    def tell(self, result: Result) -> None:
        """Tell SMAC the result of the query."""
        from smac.runhistory import StatusType, TrialValue

        match self.problem.objectives:
            case Mapping():
                cost = [
                    obj.as_minimize(result.values[key])
                    for key, obj in self.problem.objectives.items()
                ]
            case _:
                raise TypeError("Objective must be a Mapping for multi-objective problems")

        self.optimizer.tell(
            result.query.optimizer_info,  # type: ignore
            TrialValue(
                cost=cost,
                time=0.0,
                starttime=0.0,
                endtime=0.0,
                status=StatusType.SUCCESS,
                additional_info={},
            ),
            save=True,
        )
        self._initial_pending.discard(id(result.query.optimizer_info))
