"""Script to interface with the HW-GPT-Bench library."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

from momfpriors.constants import DEFAULT_DATA_DIR
from momfpriors.utils import is_package_installed

hwgpt_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hpoglue.query import Query
    from hwgpt.api import HWGPT


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

metrics = [
    "perplexity",
    "flops",
    "params",
    "bfloat16_memory",
    "float16_memory",
]

hwmetrics = [
    "energies",
    "latencies"
]

device_list = [
    "a100",
    "a40",
    "h100",
    "rtx2080",
    "rtx3080",
    "a6000",
    "v100",
    "P100",
    "cpu_xeon_silver",
    "cpu_xeon_gold",
    "cpu_amd_7502",
    "cpu_amd_7513",
    "cpu_amd_7452",
]

fidelity_maps = {
    "s": [10, 11, 12],
    "m": [22, 23, 24],
    "l": [34, 35, 36],
}

def _get_hwgptbench_config_space(
    data_dir: Path,
    space_size: Literal["s", "m", "l"],
) -> dict[str, list]:
    """Get the configuration space for the HW-GPT-Bench benchmark.

    Args:
        data_dir: The directory where the benchmark data is stored.
        space_size: The size of the search space. One of "s", "m", or "l".

    Returns:
        The configuration space for the HW-GPT-Bench benchmark.
    """
    from hwgpt.api import HWGPT  # noqa: PLC0415
    space = HWGPT(
        search_space=space_size,
        use_supernet_surrogate=False,
        base_path=data_dir,
    ).search_space
    modified_space = {}
    modified_space["embed_dim_choices"] = space["embed_dim_choices"]
    modified_space["bias_choices"] = space["bias_choices"]
    for i in range(max(space["n_layer_choices"])):
        modified_space[f"mlp_ratio_{i}"] = space["mlp_ratio_choices"]
        modified_space[f"num_heads_{i}"] = space["n_head_choices"]
    return modified_space


def _choices_to_sampled_config(
    arch_config: dict[str, int],
    fid_value: int,
) -> dict[str, list | int]:
    sampled_config = {}
    sampled_config["sample_embed_dim"] = arch_config["embed_dim_choices"]
    sampled_config["sample_n_layer"] = fid_value
    sampled_config["sample_mlp_ratio"] = []
    sampled_config["sample_n_head"] = []
    for i in range(fid_value):
        sampled_config["sample_mlp_ratio"].append(arch_config[f"mlp_ratio_{i}"])
        sampled_config["sample_n_head"].append(arch_config[f"num_heads_{i}"])
    sampled_config["sample_bias"] = str(arch_config["bias_choices"])

    return sampled_config


def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    space_size: Literal["s", "m", "l"] = "s",
    use_supernet_surrogate: bool = False,
    datadir: Path | str | None = None,
    device=None,
) -> SurrogateBenchmark:
    """Creates a SurrogateBenchmark from HW-GPT-Bench."""
    from hwgpt.api import HWGPT  # noqa: PLC0415

    if datadir is not None:
        datadir = Path(datadir).absolute().resolve()
    bench = HWGPT(
        search_space=space_size,
        use_supernet_surrogate=use_supernet_surrogate,
        base_path=datadir,
    )
    predictor = "supernet" if use_supernet_surrogate else "mlp"

    assert device is None or device in device_list, (
        f"Device {device} not in {device_list}"
    )

    query_function = partial(
        _hwgptbench_surrogate_query_function,
        benchmark=bench,
        predictor=predictor,
        device=device,
        space_size=space_size,
    )
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.search_space,
        query=query_function,
    )


def _hwgptbench_surrogate_query_function(
        query: Query,
        benchmark: HWGPT,
        space_size: Literal["s", "m", "l"] = "s",
        predictor: Literal["mlp", "supernet"] = "mlp",
        device=None,
    ) -> Result:
    from hwgpt.predictors.hwmetric.models.autogluon.autogluon_latencies import (  # noqa: PLC0415
        MultilabelPredictor,  # noqa: F401
    )
    if query.fidelity is not None:
        assert isinstance(query.fidelity, tuple)
        _, fid_value = query.fidelity
    else:
        fid_value = 3 # The max in our space is 3
    fid_value = fidelity_maps[space_size][fid_value - 1]
    config_vals = query.config.values
    sampled_config = _choices_to_sampled_config(
        arch_config=config_vals,
        fid_value=fid_value,
    )
    benchmark.set_arch(sampled_config)
    all_results = {}
    for metric in metrics:
        all_results[metric] = benchmark.query(
            metric=metric,
            predictor=predictor,
            device=device
        )
    hw_results = benchmark.query(
        device=device,
    )
    hw_results_flat = {}
    for obj, device_metric in hw_results.items():
        for device_name, value in device_metric.items():
            hw_results_flat[f"{device_name}_{obj}"] = value
    all_results.update(hw_results_flat)
    return Result(
        query=query,
        values=all_results,
        fidelity=query.fidelity,
    )


def hwgpt_benchmarks(
    datadir: Path | None = None,
    **kwargs,
) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the HW-GPT-Bench Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

        kwargs: Additional keyword arguments to pass to the HWGPT constructor.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each HW-GPT-Bench benchmark.
    """
    if datadir is None:
        datadir = DEFAULT_DATA_DIR / "HW-GPT-Bench"
    elif isinstance(datadir, str):
        datadir = Path(datadir).absolute().resolve()

    use_supernet_surrogate = kwargs.get("use_supernet_surrogate", False)
    device = kwargs.get("device")
    all_latencies = {
        f"{device}_latencies": Measure.metric(bounds=(0, np.inf), minimize=True)
        for device in device_list
    }

    all_energies = {
        f"{device}_energies": Measure.metric(bounds=(0, np.inf), minimize=True)
        for device in device_list
    }

    main_metrics = {
        "perplexity": Measure.metric(bounds=(0, np.inf), minimize=True)
    }
    main_metrics.update(all_latencies)
    main_metrics.update(all_energies)

    env = Env(
        name="py310-hwgptbench-0.1",
        python_version="3.10",
        requirements=("hwgptbench",),
        post_install=None,
    )
    for req in env.requirements:
        if not is_package_installed(req):
            hwgpt_logger.error(
                f"Please install the required package for hwgptbench: {req}",
                stacklevel=2
            )
            return
    yield BenchmarkDescription(
        name="hwgptbench-s",
        config_space=_get_hwgptbench_config_space(
            data_dir=datadir,
            space_size="s",
        ),
        load=partial(
            _get_surrogate_benchmark,
            space_size="s",
            use_supernet_surrogate=use_supernet_surrogate,
            datadir=datadir,
            device=device,
        ),
        metrics=main_metrics,
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_layer_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=False)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="hwgptbench-m",
        config_space=_get_hwgptbench_config_space(
            data_dir=datadir,
            space_size="m",
        ),
        load=partial(
            _get_surrogate_benchmark,
            space_size="m",
            use_supernet_surrogate=use_supernet_surrogate,
            datadir=datadir,
            device=device,
        ),
        metrics=main_metrics,
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_layer_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=False)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="hwgptbench-l",
        config_space=_get_hwgptbench_config_space(
            data_dir=datadir,
            space_size="l",
        ),
        load=partial(
            _get_surrogate_benchmark,
            space_size="l",
            use_supernet_surrogate=use_supernet_surrogate,
            datadir=datadir,
            device=device,
        ),
        metrics=main_metrics,
        test_metrics={},
        costs={
            "flops": Measure.metric(bounds=(0, np.inf), minimize=True),
            "params": Measure.metric(bounds=(0, np.inf), minimize=True),
            "bfloat16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
            "float16_memory": Measure.metric(bounds=(0, np.inf), minimize=True),
        },
        fidelities={
            "n_layer_choices": RangeFidelity.from_tuple((1, 3, 1), supports_continuation=False)
        },
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
