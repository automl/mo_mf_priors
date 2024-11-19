from __future__ import annotations

from momfpriors.benchmarks.botorch_momf import MOMFBC_Bench, MOMFPark_Bench

BENCHMARKS = {
    MOMFBC_Bench.desc.name: MOMFBC_Bench,
    MOMFPark_Bench.desc.name: MOMFPark_Bench,
}