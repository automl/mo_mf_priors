from hpoglue import Config, Optimizer, Problem, Query, Result
from pathlib import Path


class RandomSearch(Optimizer):
    name = "RandomSearch"
    support = Problem.Support()
    def __init__(
        self,
        problem: Problem,
        working_directory: str | Path,
        seed: int | None = None,
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