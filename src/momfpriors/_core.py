from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hpoglue._run import _trial_budget_cost

from momfpriors.constants import DEFAULT_RESULTS_DIR
from momfpriors.optimizer import Abstract_AskTellOptimizer, Abstract_NonAskTellOptimizer

if TYPE_CHECKING:
    from hpoglue import Problem, Result

    from momfpriors.csprior import CSPrior

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)


def _core(
    problem: Problem,
    priors: Mapping[str, CSPrior] | None = None,
    seed: int = 0,
    num_iterations: int = 1000,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    **kwargs: Any,
) -> list[Result] | None:


    benchmark = problem.benchmark.load(problem.benchmark)
    optimizer = problem.optimizer

    logger.info(
        f"Running {problem.optimizer.name} on {problem.benchmark.name}"
        f" with objectives {problem.get_objectives()}"
    )
    optimizer_kwargs = copy.deepcopy(problem.optimizer_hyperparameters)
    optimizer_kwargs["priors"] = priors

    if issubclass(optimizer, Abstract_NonAskTellOptimizer):
        opt = optimizer(
            problem=problem,
            seed=seed,
            working_directory=results_dir,
            **optimizer_kwargs
        )
        opt.optimize(kwargs)  # noqa: RET503

    elif issubclass(optimizer, Abstract_AskTellOptimizer):
        opt = optimizer(
            problem=problem,
            seed=seed,
            working_directory=results_dir/"Optimizers_cache",
            **optimizer_kwargs
        )
        _history: list[Result] = []
        used_budget: float = 0
        for i in range(num_iterations):
            query = opt.ask()
            result = benchmark.query(query)
            opt.tell(result)
            budget_cost = _trial_budget_cost(
                value=result.fidelity,
                problem=problem,
                minimum_normalized_fidelity=1,
            )

            used_budget += budget_cost
            result.budget_cost = budget_cost
            result.budget_used_total = used_budget
            _history.append(result)
            logger.info(f"Iteration {i+1}/{num_iterations}: {result.values}")
        return _history

    else:
        raise TypeError(
            f"Unknown optimizer type {optimizer}"
            f"Expected {Abstract_NonAskTellOptimizer} or"
            f"{Abstract_AskTellOptimizer}"
        )