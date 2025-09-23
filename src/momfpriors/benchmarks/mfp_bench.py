"""Script to interface with the MF Prior Bench library."""
# TODO(eddiebergman): Right now it's not clear how to set defaults for multi-objective.
# Do we want to prioritize obj and cost (i.e. accuracy and time) or would we rather two
# objectives (i.e. accuracy and cross entropy)?
# Second, for a benchmark to be useful, it should provide a reference point from which to compute
# hypervolume. For bounded costs this is fine but we can't do so for something like time.
# For tabular datasets, we could manually look for the worst time value
# TODO(eddiebergman): Have not included any of the conditional benchmarks for the moment
# as it seems to crash
# > "nb301": NB301Benchmark,
# > "rbv2_super": RBV2SuperBenchmark,
# > "rbv2_aknn": RBV2aknnBenchmark,
# > "rbv2_glmnet": RBV2glmnetBenchmark,
# > "rbv2_ranger": RBV2rangerBenchmark,
# > "rbv2_rpart": RBV2rpartBenchmark,
# > "rbv2_svm": RBV2svmBenchmark,
# > "rbv2_xgboost": RBV2xgboostBenchmark,
# > "iaml_glmnet": IAMLglmnetBenchmark,
# > "iaml_ranger": IAMLrangerBenchmark,
# > "iaml_rpart": IAMLrpartBenchmark,
# > "iaml_super": IAMLSuperBenchmark,
# > "iaml_xgboost": IAMLxgboostBenchmark,

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

from momfpriors.utils import is_package_installed

mfp_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mfpbench
    from hpoglue.query import Query


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:
    import mfpbench

    from momfpriors.utils import HiddenPrints

    with HiddenPrints():        # NOTE: To stop yahpo-lcbench from printing garbage
        if datadir is not None:
            datadir = Path(datadir).absolute().resolve()
            kwargs["datadir"] = datadir
        bench = mfpbench.get(benchmark_name, **kwargs)
    query_function = partial(_mfpbench_surrogate_query_function, benchmark=bench)
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.space,
        query=query_function,
    )


def _mfpbench_surrogate_query_function(query: Query, benchmark: mfpbench.Benchmark) -> Result:
    if query.fidelity is not None:
        assert isinstance(query.fidelity, tuple)
        _, fid_value = query.fidelity
    else:
        fid_value = None
    return Result(
        query=query,
        values=benchmark.query(
            query.config.values,
            at=fid_value,
        ).as_dict(),
        fidelity=query.fidelity,
    )


def _download_data_cmd(key: str, datadir: Path | None = None) -> tuple[str, ...]:
    install_cmd = f"python -m mfpbench download --benchmark {key}"
    if datadir is not None:
        install_cmd += f" --data-dir {datadir.resolve()}"
    return tuple(install_cmd.split(" "))


_lcbench_task_ids = (
    "3945",
    "7593",
    "34539",
    "126025",
    "126026",
    "126029",
    "146212",
    "167104",
    "167149",
    "167152",
    "167161",
    "167168",
    "167181",
    "167184",
    "167185",
    "167190",
    "167200",
    "167201",
    "168329",
    "168330",
    "168331",
    "168335",
    "168868",
    "168908",
    "168910",
    "189354",
    "189862",
    "189865",
    "189866",
    "189873",
    "189905",
    "189906",
    "189908",
    "189909",
)


def lcbench_surrogate(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the LCBench surrogate Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in the LCBench surrogate Benchmark.
    """
    if datadir is not None and "yahpo" in os.listdir(datadir):
        datadir = datadir / "yahpo"
    import mfpbench

    from momfpriors.utils import HiddenPrints

    env = Env(
        name="py310-mfpbench-1.10-yahpo",
        requirements=(
            "mf-prior-bench==1.10.0",
            # "yahpo-gym==1.0.2",   # Use editable install
            "xgboost>=1.7"
        ),
        post_install=_download_data_cmd("yahpo", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.error(
                f"Please install the required package for yahpo-lcbench: {req}",
                stacklevel=2
            )
            return
    with HiddenPrints():        # NOTE: To stop yahpo-lcbench from printing garbage
        for task_id in _lcbench_task_ids:
            yield BenchmarkDescription(
                name=f"yahpo-lcbench-{task_id}",
                config_space=mfpbench.get(
                    "lcbench",
                    task_id=task_id,
                    datadir=datadir,
                ).space,
                load=partial(
                    _get_surrogate_benchmark,
                    benchmark_name="lcbench",
                    datadir=datadir,
                    task_id=task_id
                ),
                metrics={
                    "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                    "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                    "val_balanced_accuracy": Measure.metric((0, 1), minimize=False),
                },
                test_metrics={
                    "test_balanced_accuracy": Measure.test_metric((0, 1), minimize=False),
                    "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
                },
                costs={
                    "time": Measure.cost((0, np.inf), minimize=True),
                },
                fidelities={
                    "epoch": RangeFidelity.from_tuple((1, 52, 1), supports_continuation=True),
                },
                env=env,
                mem_req_mb=4096,
            )


def mfh() -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the MF-Hartmann Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each combination of correlation and dimensions in the MFH Benchmarks.
    """
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.10-mfh",
        python_version="3.10",
        requirements=("mf-prior-bench==1.10.0",),
        post_install=(),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.error(f"Please install the required package for mfh: {req}", stacklevel=2)
            return
    for correlation in ("bad", "good", "moderate", "terrible"):
        for dims in (3, 6):
            name = f"mfh{dims}_{correlation}"
            _min = -3.32237 if dims == 3 else -3.86278  # noqa: PLR2004
            yield BenchmarkDescription(
                name=name,
                config_space=mfpbench.get(name).space,
                load=partial(_get_surrogate_benchmark, benchmark_name=name),
                costs={
                    "fid_cost": Measure.cost((0.05, 1), minimize=True),
                },
                fidelities={
                    "z": RangeFidelity.from_tuple((1, 100, 1), supports_continuation=True),
                },
                metrics={
                    "value": Measure.metric((_min, np.inf), minimize=True),
                },
                has_conditionals=False,
                is_tabular=False,
                env=env,
                mem_req_mb = 1024,
            )


def jahs(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the JAHSBench Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in JAHSBench.
    """
    if datadir is not None and "jahs" in os.listdir(datadir):
        datadir = datadir / "jahs"
    import mfpbench
    task_ids = ("CIFAR10", "ColorectalHistology", "FashionMNIST")
    env = Env(
        name="py310-mfpbench-1.10-jahs",
        python_version="3.10",
        requirements=(
            "mf-prior-bench==1.10.0",
            "jahs-bench==1.2.0",
            # "pandas<1.4",
            # "ConfigSpace<=0.6.1",
        ),
        post_install=_download_data_cmd("jahs", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.error(f"Please install the required package for jahs: {req}", stacklevel=2)
            return
    for task_id in task_ids:
        name = f"jahs-{task_id}"
        yield BenchmarkDescription(
            name=name,
            config_space=mfpbench.get(
                "jahs",
                task_id=task_id,
                datadir=datadir,
            ).space,
            load=partial(
                _get_surrogate_benchmark,
                benchmark_name="jahs",
                task_id=task_id,
                datadir=datadir,
            ),
            metrics={
                "valid_acc": Measure.metric((0.0, 100.0), minimize=False),
            },
            test_metrics={
                "test_acc": Measure.test_metric((0.0, 100.0), minimize=False),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 200, 1), supports_continuation=True),
            },
            costs={
                "runtime": Measure.cost((0, np.inf), minimize=True),
            },
            has_conditionals=False,
            is_tabular=False,
            env=env,
            mem_req_mb=12288,
        )


def pd1(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the PD1 Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each PD1 benchmark.
    """
    if datadir is not None and "pd1" in os.listdir(datadir):
        datadir = datadir / "pd1"
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.10-pd1",
        python_version="3.10",
        requirements=("mf-prior-bench==1.10.0","xgboost>=1.7"),
        post_install=_download_data_cmd("pd1", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.error(f"Please install the required package for pd1: {req}", stacklevel=2)
            return
    yield BenchmarkDescription(
        name="pd1-cifar100-wide_resnet-2048",
        config_space=mfpbench.get("cifar100_wideresnet_2048", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="cifar100_wideresnet_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 199, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="pd1-imagenet-resnet-512",
        config_space=mfpbench.get("imagenet_resnet_512", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="imagenet_resnet_512", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 99, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="pd1-lm1b-transformer-2048",
        config_space=mfpbench.get("lm1b_transformer_2048", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="lm1b_transformer_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 74, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )
    yield BenchmarkDescription(
        name="pd1-translate_wmt-xformer_translate-64",
        config_space=mfpbench.get("translatewmt_xformer_64", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="translatewmt_xformer_64", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 19, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )


def mfpbench_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for various MF-Prior-Bench.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each benchmark.
    """
    if datadir is None:
        datadir=Path(__file__).parent.parent.parent.parent.absolute() / "data"
    yield from lcbench_surrogate(datadir)
    yield from mfh()
    yield from jahs(datadir)
    yield from pd1(datadir)
