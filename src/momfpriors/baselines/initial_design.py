from __future__ import annotations

from copy import deepcopy
from typing import Literal

from ConfigSpace import Configuration, ConfigurationSpace, Constant


class RandomInitialDesign:
    """The first random draws from an isolated copy of the benchmark's space.

    Sample one configuration at a time, exactly like Random Search. Preserve
    repeated configurations and their order, including in finite spaces.
    """

    def __init__(
        self,
        config_space: ConfigurationSpace,
        *,
        seed: int,
        size: int | Literal["ndim"] = "ndim",
        budget: int | float,
    ) -> None:
        """Sample the Random Search prefix, capped only by the trial budget."""
        if size == "ndim":
            size = max(1, sum(not isinstance(hp, Constant) for hp in config_space.values()))
        if type(size) is not int or size < 1:
            raise ValueError("initial_design_size must be 'ndim' or a positive integer.")
        if budget < 1:
            raise ValueError("The budget must allow at least one initial evaluation.")

        self.config_space = deepcopy(config_space)
        self.config_space.seed(seed)
        self.configurations: list[Configuration] = [
            self.config_space.sample_configuration() for _ in range(int(min(size, budget)))
        ]
