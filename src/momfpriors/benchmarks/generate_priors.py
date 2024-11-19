from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hpoglue import Config, FunctionalBenchmark, Query, Result, run_glue
from momfpriors.benchmarks import BENCHMARKS
from momfpriors.benchmarks.utils import bench_first_fid, cs_random_sampling, find_incumbent
from momfpriors.optimizers.random_search import RandomSearch


def generate_priors_wrt_obj(
    seed: int,
    nsamples: int,
    to: Path,
    benchmarks: (
        Mapping[str, str]
        | Mapping[FunctionalBenchmark, str]
        | list[Mapping[str, str]]
        | list[Mapping[FunctionalBenchmark, str]]
    ),
    fidelity: int | float | None = None,
    clean: bool = False,
) -> None:
    """Generate priors for the given benchmarks.

    Args:
        seed (int): The seed to use for generating the priors.

        nsamples (int): The number of samples to generate.

        to (Path): The path to save the priors.

        benchmarks (
            Mapping[str, str]
            | Mapping[FunctionalBenchmark, str]
            | list[Mapping[str, str]]
            | list[Mapping[FunctionalBenchmark, str]]
        ): The benchmarks to generate priors for.
        Each Mapping should have the benchmark as the key and the objective to generate the priors for as the value.

        fidelity (int | float | None, optional): The fidelity to use for the benchmarks. Defaults to None.
        Only uses the first available fidelity for the benchmark.

        clean (bool, optional): Whether to clean the priors directory before generating the priors. Defaults to False.
    """
    # if to.exists() and clean:
    #     for child in filter(lambda path: path.is_file(), to.iterdir()):
    #         child.unlink()

    # to.mkdir(exist_ok=True)

    if isinstance(benchmarks, dict):
        benchmarks = [benchmarks]

    for benchmk in benchmarks:
        benchmark, objective = next(iter(benchmk.items()))
        if isinstance(benchmark, str):
            assert benchmark in BENCHMARKS, f"Unknown benchmark: {benchmark}"
            benchmark = BENCHMARKS[benchmark]

        if isinstance(benchmark, FunctionalBenchmark):
            benchmark = benchmark.description

        print(f"Generating priors for benchmark: {benchmark.name} and objective: {objective}")


        max_fidelity = bench_first_fid(benchmark).max

        if fidelity is not None:
            if (
                isinstance(max_fidelity, int)
                and isinstance(fidelity, float)
                and fidelity.is_integer()
            ):
                fidelity = int(fidelity)

            if type(fidelity) is not type(max_fidelity):
                raise ValueError(
                    f"Cannot use fidelity {fidelity} (type={type(fidelity)}) with"
                    f" benchmark {benchmark.name}",
                )
            at = fidelity
        else:
            at = max_fidelity

        # results: list[Result] = []
        # bench = benchmark.load(benchmark)
        # for query in cs_random_sampling(
        #     benchmark=benchmark,
        #     nsamples=nsamples,
        #     seed=seed,
        #     at=at,
        # ):
        #     results.append(bench.query(query))

        _df = run_glue.run_glue(
            optimizer=RandomSearch,
            benchmark=benchmark,
            seed=seed,
            budget=nsamples
        )

        print(_df)


if __name__ == "__main__":
    generate_priors_wrt_obj(
        seed=1,
        nsamples=10,
        to=Path("priors"),
        benchmarks=[
            {"MOMFPark": "value1"},
            {"MOMFPark": "value2"},
        ],
        fidelity=100,
        clean=True
    )
