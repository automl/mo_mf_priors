from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from hpoglue import FunctionalBenchmark, Result

from momfpriors.benchmarks import BENCHMARKS
from momfpriors.benchmarks.utils import bench_first_fid, cs_random_sampling, get_prior_configs


def generate_priors_wrt_obj(
    seed: int,
    nsamples: int,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]],
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
        seed: The seed to use for generating the priors.

        nsamples: The number of samples to generate.

        prior_spec: The prior specification to use for generating the priors.

        to: The path to save the priors.

        benchmarks: The benchmarks to generate priors for.
            Each Mapping should have the benchmark as the key and the objective to
            generate the priors for as the value.

        fidelity: The fidelity to use for the benchmarks.
            Defaults to None.
            Only uses the first available fidelity for the benchmark.

        clean: Whether to clean the priors directory before generating the priors.
            Defaults to False.
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

        results: list[Result] = []
        bench = benchmark.load(benchmark)
        for query in cs_random_sampling(
            benchmark=benchmark,
            nsamples=nsamples,
            seed=seed,
            at=at,
        ):
            results.append(bench.query(query))

        prior_spec_results = []
        _results = sorted(results, key=lambda r: r.values[objective])

        for _, index, _, _ in prior_spec:
            prior_spec_results.append(_results[index])

        print(".\n".join([str(res) for res in prior_spec_results]))

        prior_configs = get_prior_configs(
            results=results,
            space=benchmark.config_space,
            objective=objective,
            seed=seed,
            prior_spec=prior_spec,
        )

        print(prior_configs)



if __name__ == "__main__":
    generate_priors_wrt_obj(
        seed=1,
        nsamples=10,
        prior_spec=[
            ("good", 0, 0.01, None)
        ],
        to=Path("priors"),
        benchmarks=[
            {"MOMFBraninCurrin": "value1"},
            {"MOMFBraninCurrin": "value2"},
        ],
        fidelity=100,
        clean=True
    )
