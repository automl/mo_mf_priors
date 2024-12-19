from __future__ import annotations

import logging
import shutil
import traceback
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd
import yaml
from hpoglue.env import (
    GLUE_PYPI,
    Env,
    Venv,
    get_current_installed_hpoglue_version,
)
from hpoglue.problem import Problem

from momfpriors._run import _run

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription
    from hpoglue.result import Result

    from momfpriors.optimizer import Abstract_AskTellOptimizer, Abstract_NonAskTellOptimizer


T = TypeVar("T", bound=Hashable)


logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def _try_delete_if_exists(path: Path) -> None:
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error deleting {path}: {e}")



@dataclass
class Run:
    """A run of a benchmark."""

    problem: Problem
    """The problem that was run."""

    seed: int
    """The seed used for the run."""

    optimizer: type[Abstract_AskTellOptimizer | Abstract_NonAskTellOptimizer]
    """The optimizer to use for the Problem"""

    rootdir: Path = field(init=False)
    """Default root directory."""

    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Extra kwargs for the optimizer."""

    benchmark: BenchmarkDescription
    """The benchmark that the Problem was run on."""

    env: Env = field(init=False)
    """The environment to setup the optimizer in for `isolated` mode."""

    working_dir: Path = field(init=False)
    """The working directory for the run."""

    complete_flag: Path = field(init=False)
    """The flag to indicate the run is complete."""

    error_file: Path = field(init=False)
    """The file to store the error traceback if the run crashed."""

    running_flag: Path = field(init=False)
    """The flag to indicate the run is currently running."""

    queue_flag: Path = field(init=False)
    """The flag to indicate the run is queued."""

    df_path: Path = field(init=False)
    """The path to the dataframe for the run."""

    env_path: Path = field(init=False)
    """The path to the environment for the run."""

    mem_req_mb: int = field(init=False)
    """The memory requirement for the run in mb.
    Calculated as the sum of the memory requirements of the optimizer and the benchmark.
    """

    def __post_init__(self) -> None:
        self.benchmark = self.problem.benchmark
        self.optimizer = self.problem.optimizer
        self.optimizer_hyperparameters = self.problem.optimizer_hyperparameters
        self.mem_req_mb = self.problem.mem_req_mb

        name_parts: list[str] = [
            self.problem.name,
            f"seed={self.seed}",
        ]
        self.name = ".".join(name_parts)
        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self.problem)

        match self.benchmark.env, self.optimizer.env:
            case (None, None):
                self.env = Env.empty()
            case (None, Env()):
                self.env = self.optimizer.env
            case (Env(), None):
                self.env = self.benchmark.env
            case (Env(), Env()):
                self.env = Env.merge(self.benchmark.env, self.optimizer.env)
            case _:
                raise ValueError("Invalid combination of benchmark and optimizer environments")


    def _set_paths(self, expdir: Path) -> None:
        self.expdir = expdir
        self.working_dir = self.expdir.absolute().resolve() / self.name
        self.complete_flag = self.working_dir / "complete.flag"
        self.error_file = self.working_dir / "error.txt"
        self.running_flag = self.working_dir / "running.flag"
        self.df_path = self.working_dir / f"{self.name}.parquet"
        self.venv_requirements_file = self.working_dir / "venv_requirements.txt"
        self.queue_flag = self.working_dir / "queue.flag"
        self.requirements_ran_with_file = self.working_dir / "requirements_ran_with.txt"
        self.env_description_file = self.working_dir / "env.yaml"
        self.post_install_steps = self.working_dir / "venv_post_install.sh"
        self.run_yaml_path = self.working_dir / "run_config.yaml"
        self.env_path = self.expdir / "envs" / self.env.identifier


    @property
    def venv(self) -> Venv:
        return Venv(self.env_path)

    @property
    def conda(self) -> Venv:
        raise NotImplementedError("Conda not implemented yet.")

    def run(
        self,
        *,
        on_error: Literal["raise", "continue"] = "raise",
    ) -> pd.DataFrame:
        """Run the Run.

        Args:
            on_error: How to handle errors. In any case, the error will be written
                into the [`working_dir`][hpoglue.run.Run.working_dir]
                of the problem.

                * If "raise", raise an error.
                * If "continue", log the error and continue.
        """
        if on_error not in ("raise", "continue"):
            raise ValueError(f"Invalid value for `on_error`: {on_error}")


        if self.df_path.exists():
            logger.info(f"Loading results for {self.name} from {self.working_dir}")
            return Run.Report.from_df(
                df=pd.read_parquet(self.df_path),
                run=self,
            )

        """ TODO
        if self.working_dir.exists():
            raise RuntimeError(
                "The optimizer ran before but no dataframe of results was found at "
                f"{self.df_path}."
            )
        """
        self.set_state(Run.State.PENDING)
        _hist: list[Result] = []
        try:

            self.set_state(self.State.RUNNING)
            _hist = _run(
                problem=self.problem,
                seed=self.seed,
                run_name=self.name,
                on_error="raise",
            )
        except Exception as e:
            self.set_state(Run.State.CRASHED, err_tb=(e, traceback.format_exc()))
            logger.exception(e)
            logger.error(f"Error in Run {self.name}: {e}")
            match on_error:
                case "raise":
                    raise e
                case "continue":
                    raise NotImplementedError("Continue not yet implemented!") from e
                case _:
                    raise RuntimeError(f"Invalid value for `on_error`: {on_error}") from e
        logger.info(f"COMPLETED running {self.name}")
        logger.info(f"Saving {self.name} at {self.working_dir}")
        _df = self.create_df(history=_hist)
        logger.info(f"Results dumped at {self.df_path.absolute()}")
        return _df


    def create_df(
        self,
        *,
        history: list[Result],
    ) -> pd.DataFrame:
        """Create a dataframe from the history."""
        _df = pd.DataFrame([r.as_dict() for r in history])
        _df.to_parquet(self.df_path, index=False)
        self.set_state(Run.State.COMPLETE, df=_df())
        return _df


    def create_env(
        self,
        *,
        how: Literal["venv", "conda"] = "venv",
        hpoglue: Literal["current_version"] | str,
    ) -> None:
        """Set up the isolation for the experiment."""
        if hpoglue == "current_version":
            raise NotImplementedError("Not implemented yet.")

        match hpoglue:
            case "current_version":
                _version = get_current_installed_hpoglue_version()
                req = f"{GLUE_PYPI}=={_version}"
            case str():
                req = hpoglue
            case _:
                raise ValueError(f"Invalid value for `hpoglue`: {hpoglue}")

        requirements = [req, *self.env.requirements]

        logger.info(f"Installing deps: {self.env.identifier}")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        with self.venv_requirements_file.open("w") as f:
            f.write("\n".join(requirements))

        if self.env_path.exists():
            return

        self.env_path.parent.mkdir(parents=True, exist_ok=True)

        env_dict = self.env.to_dict()
        env_dict.update({"env_path": str(self.env_path), "hpoglue_source": req})

        logger.info(f"Installing env: {self.env.identifier}")
        match how:
            case "venv":
                logger.info(f"Creating environment {self.env.identifier} at {self.env_path}")
                self.venv.create(
                    path=self.env_path,
                    python_version=self.env.python_version,
                    requirements_file=self.venv_requirements_file,
                    exists_ok=False,
                )
                if self.env.post_install:
                    logger.info(f"Running post install for {self.env.identifier}")
                    with self.post_install_steps.open("w") as f:
                        f.write("\n".join(self.env.post_install))
                    self.venv.run(self.env.post_install)
            case "conda":
                raise NotImplementedError("Conda not implemented yet.")
            case _:
                raise ValueError(f"Invalid value for `how`: {how}")


    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "seed": self.seed,
            "expdir": str(self.expdir),
        }

    @classmethod
    def from_yaml(cls, path: Path) -> Run:
        with path.open("r") as file:
            return Run.from_dict(yaml.safe_load(file))

    def write_yaml(self) -> None:
        self.run_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_yaml_path.open("w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:

        return Run(
            problem=Problem.from_dict(data["problem"]),
            seed=data["seed"],
            expdir=Path(data["expdir"]),
        )

    def state(self) -> Run.State:
        """Return the state of the run.

        Args:
            run: The run to get the state for.
        """
        if self.complete_flag.exists():
            return Run.State.COMPLETE

        if self.error_file.exists():
            return Run.State.CRASHED

        if self.running_flag.exists():
            return Run.State.RUNNING

        if self.queue_flag.exists():
            return Run.State.QUEUED

        return Run.State.PENDING

    def set_state(  # noqa: C901, PLR0912
        self,
        state: Run.State,
        *,
        df: pd.DataFrame | None = None,
        err_tb: tuple[Exception, str] | None = None,
    ) -> None:
        """Set the run to a certain state.

        Args:
            state: The state to set the problem to.
            df: Optional dataframe to save if setting to [`Run.State.COMPLETE`].
            err_tb: Optional error traceback to save if setting to [`Run.State.CRASHED`].
        """
        _flags = (self.complete_flag, self.error_file, self.running_flag, self.queue_flag)
        match state:
            case Run.State.PENDING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                with self.run_yaml_path.open("w") as f:
                    yaml.dump(self.to_dict(), f, sort_keys=False)

            case Run.State.QUEUED:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.queue_flag.touch()

            case Run.State.RUNNING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.df_path.parent.mkdir(parents=True, exist_ok=True)

                # lines = subprocess.run(
                #     [self.venv.pip, "freeze"],  # noqa: S603
                #     check=True,
                #     capture_output=True,
                #     text=True,
                # )
                # with self.requirements_ran_with_file.open("w") as f:
                #     f.write(lines.stdout)

                self.running_flag.touch()

            case Run.State.CRASHED:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                with self.error_file.open("w") as f:
                    if err_tb is None:
                        f.write("None")
                    else:
                        exc, tb = err_tb
                        f.write(f"{tb}\n{exc}")

            case Run.State.COMPLETE:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                self.complete_flag.touch()

                if df is not None:
                    df.to_parquet(self.df_path)
            case _:
                raise ValueError(f"Unknown state {state}")


    class State(str, Enum):
        """The state of a problem."""

        PENDING = "PENDING"
        QUEUED = "QUEUED"
        RUNNING = "RUNNING"
        CRASHED = "CRASHED"
        COMPLETE = "COMPLETE"

        @classmethod
        def collect(
            cls,
            state: str | Run.State | bool | Iterable[Run.State | str],
        ) -> list[Run.State]:
            """Collect state requested."""
            match state:
                case True:
                    return list(cls)
                case False:
                    return []
                case Run.State():
                    return [state]
                case str():
                    return [cls(state)]
                case _:
                    return [cls(s) if isinstance(s, str) else s for s in state]
