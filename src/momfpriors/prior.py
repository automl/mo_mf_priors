from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats as stats
from ConfigSpace import (
    Beta,
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    Integer,
    Normal,
    OrdinalHyperparameter,
    Uniform,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


@dataclass(kw_only=True)
class Prior:

    config_space: ConfigurationSpace

    prior_config: Mapping[str, Any]

    distribution: ConfigurationSpace = field(init=False)

    def __post_init__(self) -> None:
        self._create_prior_distribution()

    def _create_prior_distribution(self) -> ConfigurationSpace:
        self.distribution = ConfigurationSpace(seed=0)
        for hp in list(self.config_space.values()):
            default = self.prior_config[hp.name]
            distribution = Normal(default, 0.25)
            match hp:
                case UniformFloatHyperparameter():
                    self.distribution.add(
                        Float(
                            name=hp.name,
                            bounds=(hp.lower, hp.upper),
                            default=default,
                            log=hp.log,
                            distribution=distribution,
                        )
                    )
                case UniformIntegerHyperparameter():
                    self.distribution.add(
                        Integer(
                            name=hp.name,
                            bounds=(hp.lower, hp.upper),
                            default=default,
                            log=hp.log,
                            distribution=distribution,
                        )
                    )
                case CategoricalHyperparameter() | OrdinalHyperparameter():
                    hp.default_value = default
                    self.distribution.add(hp)
                case Constant():
                    self.distribution.add(hp)
                case _:
                    raise ValueError(
                        f"Unsupported hyperparameter type: {type(hp).__name__}"
                    )