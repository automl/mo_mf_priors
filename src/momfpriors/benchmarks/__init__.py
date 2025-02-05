from __future__ import annotations

# from momfpriors.benchmarks.bbob_mo import bbob_mo_benchmarks
from momfpriors.benchmarks.botorch_momf import MOMFBC_Bench, MOMFPark_Bench
from momfpriors.benchmarks.mfp_bench import mfpbench_benchmarks
# from momfpriors.benchmarks.hpobench import hpobench_benchmarks

BENCHMARKS = {
    MOMFBC_Bench.desc.name: MOMFBC_Bench,
    MOMFPark_Bench.desc.name: MOMFPark_Bench,
}

for desc in mfpbench_benchmarks():
    BENCHMARKS[desc.name] = desc

# for desc in bbob_mo_benchmarks():
#     BENCHMARKS[desc.name] = desc
# for desc in hpobench_benchmarks():
#     BENCHMARKS[desc.name] = desc

__all__ = ["BENCHMARKS"]
