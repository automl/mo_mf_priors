from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import neps
import numpy as np
import pandas as pd
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from hpoglue import BenchmarkDescription, Config, Query, Result

DEFAULT_CONFIDENCE_SCORES: Mapping[str, float] = {
    "low": 0.5,
    "medium": 0.25,
    "high": 0.125,
}


def find_incumbent(
    df: pd.DataFrame,
    results_col: str,
    objective: str,
    *,
    minimize: bool = True,
) -> float:
    """Find the best value of the objective in the results column of the dataframe.

    Args:
        df : The dataframe containing the results.

        results_col : The name of the column containing the results.

        objective : The name of the objective to minimize.

        minimize : Whether to minimize the objective. Defaults to True.

    Returns:
        The best value of the objective.
    """
    if df.empty:
        return np.nan
    vals = []
    for i in df[results_col]:
        vals.append(i[objective])
    if not minimize:
        return max(vals)
    return min(vals)

def get_prior_configs(
    results: list[Result],
    space: ConfigurationSpace,
    objective: str,
    seed: int,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]]
) -> Mapping[str, Config]:
    """Generate prior configurations based on the given results and specifications.

    Args:
        results: A list of Result objects to be used for generating prior configurations.

        space: The configuration space within which the configurations are defined.

        objective: The objective key used to sort the results.

        seed: The seed for random number generation to ensure reproducibility.

        prior_spec: An iterable of tuples specifying the prior configurations.
            Each tuple contains:
                - name: The name of the prior configuration.
                - index: The index of the result to be used.
                - std: The standard deviation for perturbation.
                - categorical_swap_chance: The probability of swapping categorical values
                    during perturbation.

    Returns:
        A mapping of prior configuration names to their perturbed configurations.
    """
    match results:
        case list():
            pass
        case _:
            raise TypeError("results must be a DataFrame or a list of Results"
        )

    prior_spec = list(prior_spec)

    print(" - Finished results")
    results = sorted(results, key=lambda r: r.values[objective])
    print(" - Finished sorting")

    prior_configs = {
            name: (results[index].query.config, std, categorical_swap_chance)
            for name, index, std, categorical_swap_chance in prior_spec
        }

    return {
            name: perturb(
                config,
                space,
                seed=seed,
                std=std,
                categorical_swap_chance=categorical_swap_chance,
            )
            for name, (config, std, categorical_swap_chance) in prior_configs.items()
        }


def bench_first_fid(benchmark: BenchmarkDescription) -> int:
    return benchmark.fidelities[
        next(iter(benchmark.fidelities.keys()))
    ]


def cs_random_sampling(
    benchmark: BenchmarkDescription,
    nsamples: int,
    seed: int,
    at: int,
    ) -> list[Config]:
    """Sample configurations from the benchmark's configuration space.

    Args:
        benchmark: The benchmark description.

        nsamples: The number of samples to draw from the configuration space.

        seed: The seed for random number generation.

        at: The fidelity value to use for the sampled configurations.

    Returns:
        The sampled configurations as a list of Config objects.
    """
    benchmark.config_space.seed(seed)
    configs = benchmark.config_space.sample_configuration(nsamples)
    return [
        Query(
            config=Config(
                config_id=None,
                values=dict(config),
            ),
            fidelity=(bench_first_fid(benchmark), at)
        )
            for config in configs
    ]


def perturb(  # noqa: C901, PLR0912
    config: Config,
    space: ConfigurationSpace,
    seed: int,
    std: float | None = None,
    categorical_swap_chance: float | None = None,
) -> Config:
    """Perturb a configuration.

    Args:
        config: The configuration to perturb.

        space: The configuration space.

        seed: The seed for random number generation.

        std: The standard deviation of the perturbation. Defaults to None.

        categorical_swap_chance: The chance of swapping a categorical value.
            Defaults to None.

    Returns:
        Config: The perturbed configuration.
    """
    perturbed_val: dict[str, int | float] = {}

    for name, value in config.values.items():
        hp = space.get(name)
        assert 0 <= std <= 1, "Noise must be between 0 and 1"
        rng: np.random.RandomState
        if seed is None:
            rng = np.random.RandomState()
        elif isinstance(seed, int):
            rng = np.random.RandomState(seed)
        else:
            rng = seed

        match hp:
            case Constant():
                _val = value

            case (
                NormalIntegerHyperparameter()
                | NormalFloatHyperparameter()
                | UniformFloatHyperparameter()
                | UniformIntegerHyperparameter()
            ):
                # TODO:
                # * https://github.com/automl/ConfigSpace/issues/287
                # * https://github.com/automl/ConfigSpace/issues/290
                # * https://github.com/automl/ConfigSpace/issues/291
                # Doesn't act as intended
                assert hp.upper is not None
                assert hp.lower is not None
                # assert hp.q is None
                assert isinstance(value, int | float)

                match hp:
                    case UniformIntegerHyperparameter():
                        if hp.log:
                            _lower = np.log(hp.lower)
                            _upper = np.log(hp.upper)
                        else:
                            _lower = hp.lower
                            _upper = hp.upper
                    case NormalIntegerHyperparameter():
                        _lower = hp.nfhp._lower
                        _upper = hp.nfhp._upper
                    case UniformFloatHyperparameter() | NormalFloatHyperparameter():
                        _lower = hp.lower_vectorized
                        _upper = hp.upper_vectorized
                    case _:
                        raise TypeError("Weird TypeError")

                space_length = std * (_upper - _lower)
                rescaled_std = std * space_length

                if not hp.log:
                    sample = np.clip(rng.normal(value, rescaled_std), _lower, _upper)
                else:
                    logged_value = np.log(value)
                    sample = rng.normal(logged_value, rescaled_std)
                    sample = np.clip(np.exp(sample), hp.lower, hp.upper)

                if isinstance(hp, UniformIntegerHyperparameter | NormalIntegerHyperparameter):
                    _val = int(np.rint(sample))  # type: ignore

                if isinstance(hp, UniformFloatHyperparameter | NormalFloatHyperparameter):
                    _val = float(sample)  # type: ignore

                perturbed_val[name] = _val

            case _:
                raise TypeError(f"Unsupported hyperparameter type: {type(hp)}")

    return Config(
        config_id=config.config_id,
        values=perturbed_val,
    )


def objective_fn_wrapper(
    objective_fn: Callable,
    **config: Mapping[str, Any]
) -> Mapping[str, Any]:
    query = Query(
        config=Config(config_id=None, values=config),
        fidelity=None,
    )
    result: Result = objective_fn(query)
    return result.values


def pipeline_space_to_cs(
    pipeline_space: Mapping[str, Any],
)-> ConfigurationSpace:
    """Convert a Pipeline Space to ConfigSpace."""
    cs = ConfigurationSpace()
    for name, param in pipeline_space.items():
        match param:
            case neps.Float():
                cs.add(
                    NormalFloatHyperparameter(
                        name=name,
                        mu= param.default,
                        sigma=DEFAULT_CONFIDENCE_SCORES[param.default_confidence_choice],
                        lower=param.lower,
                        upper=param.upper,
                        default_value=param.default,
                        log=param.log,
                    )
                )
            case neps.Integer():
                cs.add(
                    NormalIntegerHyperparameter(
                        name=name,
                        mu= param.default,
                        sigma=DEFAULT_CONFIDENCE_SCORES[param.default_confidence_choice],
                        lower=param.lower,
                        upper=param.upper,
                        default_value=param.default,
                        log=param.log,
                    )
                )
            case neps.Categorical():
                cs.add(
                    CategoricalHyperparameter(
                        name=name,
                        choices=param.choices,
                        default_value=param.default,
                    )
                )
            case neps.Constant():
                cs.add(
                    Constant(
                        name=name,
                        value=param.value,
                    )
                )
            case _:
                raise ValueError(f"Unsupported parameter type: {type(param)}")
    return cs