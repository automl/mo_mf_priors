from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from momfpriors.benchmarks.botorch_momf import MOMFBC_Bench, MOMFPark_Bench
from momfpriors.benchmarks.hwgptbench import hwgpt_benchmarks
from momfpriors.benchmarks.mf_zdt import MFZDT1Bench, MFZDT6Bench
from momfpriors.benchmarks.mfp_bench import jahs, lcbench_surrogate, pd1
from momfpriors.constants import DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from hpoglue import BenchmarkDescription, FunctionalBenchmark

all_benches: dict[str, BenchmarkDescription | FunctionalBenchmark] = {}

def gen_benches(
    datadir: str | Path | None = None,
    **kwargs,
) -> dict[str, BenchmarkDescription | FunctionalBenchmark]:
    """Generate benchmark descriptions."""
    # Main Repo directory / data
    if datadir is None:
        datadir=DEFAULT_DATA_DIR
    elif isinstance(datadir, str):
        datadir = Path(datadir)

    # LCBench
    for desc in lcbench_surrogate(datadir / "yahpo"):
        all_benches[desc.name] = desc

    # PD1
    for desc in pd1(datadir / "pd1"):
        all_benches[desc.name] = desc

    # JAHS
    for desc in jahs(datadir / "jahs"):
        all_benches[desc.name] = desc

    # HW-GPT-Bench
    for desc in hwgpt_benchmarks(datadir / "HW-GPT-Bench", **kwargs):
        all_benches[desc.name] = desc

    # Functional Benchmarks
    all_benches[MOMFBC_Bench.desc.name] = MOMFBC_Bench
    all_benches[MOMFPark_Bench.desc.name] = MOMFPark_Bench
    all_benches[MFZDT1Bench.desc.name] = MFZDT1Bench
    all_benches[MFZDT6Bench.desc.name] = MFZDT6Bench

    return dict(sorted(all_benches.items()))

BENCHMARKS = gen_benches

__all__ = ["BENCHMARKS"]
