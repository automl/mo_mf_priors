"""Script to interface with the MF Prior Bench library."""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mfpbench
import numpy as np
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

if TYPE_CHECKING:
    from hpoglue.query import Query


def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:

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
    task_ids = ("CIFAR10", "ColorectalHistology", "FashionMNIST")
    env = Env(
        name="py310-mfpbench-1.9-jahs",
        python_version="3.10",
        requirements=("mf-prior-bench[jahs-bench]==1.9.0",),
        post_install=_download_data_cmd("jahs", datadir=datadir),
    )
    for task_id in task_ids:
        name = f"jahs-{task_id}"
        yield BenchmarkDescription(
            name=name,
            config_space=mfpbench.get("jahs", task_id=task_id).space,
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
    env = Env(
        name="py310-mfpbench-1.9-pd1",
        python_version="3.10",
        requirements=("mf-prior-bench[pd1]==1.9.0",),
        post_install=_download_data_cmd("pd1", datadir=datadir),
    )
    yield BenchmarkDescription(
        name="pd1-cifar100-wide_resnet-2048",
        config_space=mfpbench.get("cifar100_wideresnet_2048", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="cifar100_wideresnet_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics=None,
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
        test_metrics=None,
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
        test_metrics=None,
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
        test_metrics=None,
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
        datadir=Path(__file__).parent.parent.parent.absolute() / "data"
    # yield from jahs(datadir)
    yield from pd1(datadir)
