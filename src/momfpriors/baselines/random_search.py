from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import Config, Problem, Query, Result

from momfpriors.optimizer import Abstract_AskTellOptimizer
from momfpriors.prior import Prior


class RandomSearch(Abstract_AskTellOptimizer):
    name = "RandomSearch"
    support = Problem.Support(
        objectives=("single", "many"),
        fidelities=(None),
        cost_awareness=(None),
        tabular=False
    )
    def __init__(
        self,
        problem: Problem,
        working_directory: str | Path,
        seed: int = 0,
        **kwargs: Any
    ):
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        self._optmizer_unique_id = 0

    def ask(self) -> Query:
        self._optmizer_unique_id += 1
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(self.config_space.sample_configuration()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:
        return


class RandomSearchWithPriors(Abstract_AskTellOptimizer):
    name = "RandomSearchWithPriors"
    support = Problem.Support(
        objectives=("single", "many"),
        fidelities=(None),
        cost_awareness=(None),
        tabular=False
    )
    def __init__(
        self,
        problem: Problem,
        working_directory: str | Path,
        priors: Mapping[str, Any] | list[Mapping[str, Any]],
        mo_prior_sampling: Literal["random", "50-50", "sequential"] = "random",
        seed: int = 0,
        **kwargs: Any
    ) -> None:
        self.config_space = problem.config_space
        self.config_space.seed(seed)
        self.problem = problem
        if isinstance(priors, Mapping):
            priors = [priors]
        self.priors = [
            Prior(
                config_space=self.config_space,
                prior_config=prior,
            )
            for prior in priors
        ]
        self.rng = np.random.default_rng(seed)
        self.mo_prior_sampling = mo_prior_sampling
        self._optmizer_unique_id = 0

    def ask(self) -> Query:
        self._optmizer_unique_id += 1
        match self.mo_prior_sampling:
            case "random":
                prior = self.rng.choice(self.priors)
            case "50-50":
                raise NotImplementedError
            case "sequential":
                raise NotImplementedError
            case _:
                raise ValueError(
                    "Invalid value for `mo_prior_sampling`. "
                    "Expected one of ['random', '50-50', 'sequential']."
                    f"Got {self.mo_prior_sampling}."
                )
        config = Config(
            config_id=str(self._optmizer_unique_id),
            values=dict(prior.distribution.sample_configuration()),
        )
        return Query(config=config, fidelity=None)

    def tell(self, result: Result) -> None:
        return