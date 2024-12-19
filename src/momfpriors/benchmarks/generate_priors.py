from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from hpoglue import FunctionalBenchmark, Result

from momfpriors.benchmarks import BENCHMARKS
from momfpriors.benchmarks.bbob_mo import bbob_function_definitions, create_bbob_mo_desc
from momfpriors.benchmarks.utils import bench_first_fid, cs_random_sampling, get_prior_configs


def generate_priors_wrt_obj(  # noqa: C901, PLR0912
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
    *,
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
    if to.exists() and clean:
        for child in filter(lambda path: path.is_file(), to.iterdir()):
            child.unlink()

    to.mkdir(exist_ok=True)

    if isinstance(benchmarks, dict):
        benchmarks = [benchmarks]

    for benchmk in benchmarks:
        benchmark, objective = next(iter(benchmk.items()))
        if isinstance(benchmark, str):
            if benchmark.startswith("bbob"):
                benchmark = create_bbob_mo_desc(func=benchmark)
            else:
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

        for name, config in prior_configs.items():
            with (to / f"{benchmark.name}_{objective}_{name}.yaml").open("w") as f:
                yaml.dump(
                    {
                        "benchmark": benchmark.name,
                        "prior_name": name,
                        "objective": objective,
                        "config": config.values,
                    }, f
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=10
    )
    parser.add_argument(
        "--prior_spec",
        nargs="+",
        type=str,
        default=[
            "good:0:0.01:None",
            "medium:0:0.125:None",
            "bad:-1:0:None",
        ],
    )
    parser.add_argument(
        "--to",
        type=Path,
        default=Path("./src/priors"),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        default=[
            "MOMFBraninCurrin:value1",
            "MOMFBraninCurrin:value2",
            "MOMFPark:value1",
            "MOMFPark:value2",
        ],
    )
    parser.add_argument(
        "--fidelity",
        type=float,
        default=None
    )
    parser.add_argument(
        "--clean",
        action="store_true"
    )
    args = parser.parse_args()

    if isinstance(args.benchmarks, str):
        args.benchmarks = [args.benchmarks]

    _benchmarks: list[Mapping[str, str]] = []
    for benchmark in args.benchmarks:
        if ":" not in benchmark:
            raise ValueError(
                "Invalid benchmark specification!"
                "Expected format: 'benchmark:objective'"
                f"Got: '{benchmark}'"
            )
        bench_name, objective = benchmark.split(":")
        _benchmarks.append({bench_name.strip(): objective.strip()}
        )

    if isinstance(args.prior_spec, str):
        args.prior_spec = [args.prior_spec]

    _prior_spec: list[tuple[str, int, float | None, float | None]] = []
    for prior in args.prior_spec:
        if ":" not in prior:
            raise ValueError(
                "Invalid prior specification!"
                "Expected format: 'name:index:std:categorical_swap_chance'"
                f"Got: '{prior}'"
            )
        name, index, std, categorical_swap_chance = prior.split(":")
        _prior_spec.append(
            (
                name.strip(),
                int(index.strip()),
                float(std.strip()) if std != "None" else None,
                float(categorical_swap_chance.strip())
                if categorical_swap_chance != "None"
                else None,
            )
        )

    generate_priors_wrt_obj(
        seed=args.seed,
        nsamples=args.nsamples,
        prior_spec=_prior_spec,
        to=args.to,
        benchmarks=_benchmarks,
        fidelity=args.fidelity,
        clean=args.clean,
    )
