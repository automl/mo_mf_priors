from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from hpoglue import Query
from hpoglue.utils import first
from neps.optimizers.mopriors import MOPriorSampler
from neps.space.encoding import ConfigEncoder
from neps.space.parsing import convert_configspace

from momfpriors.benchmarks import BENCHMARKS
from momfpriors.constants import DEFAULT_DATA_DIR, DEFAULT_PRIORS_DIR
from momfpriors.utils import cs_random_sampling

if TYPE_CHECKING:
    from hpoglue import BenchmarkDescription, Result
    from neps.sampling.priors import Prior


def rank_correlation(
    benchmark: BenchmarkDescription,
    prior_dists: dict[str, Prior],
    encoder: ConfigEncoder,
    objective1: str,
    objective2: str,
    n: int = 10,
) -> pd.DataFrame:
    """Calculate rank correlations between benchmark objectives and prior distributions.

    This function samples configurations from a benchmark, evaluates them on two objectives,
    and computes the rank correlations between the actual objective performance and the
    rankings predicted by good/bad prior distributions for each objective.

    Args:
        benchmark: The benchmark object containing metrics and costs for evaluation
        prior_dists: Dictionary containing prior distributions with keys
        encoder: Neps ConfigEncoder object used to transform configurations for prior distribution evaluation
        objective1: Name of the first objective to analyze
        objective2: Name of the second objective to analyze
        n: Number of configurations to sample and evaluate. Defaults to 10.

    Returns:
        pd.DataFrame: Spearman correlation matrix between objective ranks and prior distribution
                        ranks. Contains correlations between:
                        - obj1_rank: Ranking based on objective1 performance
                        - obj2_rank: Ranking based on objective2 performance
                        - good_prior_obj1_rank: Ranking based on good prior for objective1
                        - good_prior_obj2_rank: Ranking based on good prior for objective2
                        - bad_prior_obj1_rank: Ranking based on bad prior for objective1
                        - bad_prior_obj2_rank: Ranking based on bad prior for objective2
    """
    config_res = sample_and_eval(
        benchmark=benchmark,
        n=n,
    )
    rank_dict = {}
    bench_obj1 = benchmark.metrics.get(
        objective1,
        benchmark.costs.get(
            objective1,
        )
    )
    bench_obj2 = benchmark.metrics.get(
        objective2,
        benchmark.costs.get(
            objective2,
        )
    )
    assert bench_obj1 is not None, f"Objective {objective1} not found in benchmark"
    assert bench_obj2 is not None, f"Objective {objective2} not found in benchmark"

    for result in config_res:
        config = result.query.config.values
        perf_obj1 = bench_obj1.as_minimize(result.values[objective1])
        perf_obj2 = bench_obj2.as_minimize(result.values[objective2])
        good_prior_obj1 = prior_dists[f"{objective1}:good"].pdf_configs(
            [config], frm=encoder
        ).sum().item()
        bad_prior_obj1 = prior_dists[f"{objective1}:bad"].pdf_configs(
            [config], frm=encoder
        ).sum().item()
        good_prior_obj2 = prior_dists[f"{objective2}:good"].pdf_configs(
            [config], frm=encoder
        ).sum().item()
        bad_prior_obj2 = prior_dists[f"{objective2}:bad"].pdf_configs(
            [config], frm=encoder
        ).sum().item()
        rank_dict[result.query.config.config_id] = {
            "perf_obj1": perf_obj1,
            "perf_obj2": perf_obj2,
            "good_prior_obj1": good_prior_obj1,
            "bad_prior_obj1": bad_prior_obj1,
            "good_prior_obj2": good_prior_obj2,
            "bad_prior_obj2": bad_prior_obj2,
        }

    _df = pd.DataFrame.from_dict(rank_dict, orient="index")
    _df["obj1_rank"] = _df["perf_obj1"].rank()
    _df["obj2_rank"] = _df["perf_obj2"].rank()
    _df["good_prior_obj1_rank"] = _df["good_prior_obj1"].rank(ascending=False)
    _df["good_prior_obj2_rank"] = _df["good_prior_obj2"].rank(ascending=False)
    _df["bad_prior_obj1_rank"] = _df["bad_prior_obj1"].rank(ascending=False)
    _df["bad_prior_obj2_rank"] = _df["bad_prior_obj2"].rank(ascending=False)

    ranked_correlations = _df[[
        "obj1_rank",
        "obj2_rank",
        "good_prior_obj1_rank",
        "good_prior_obj2_rank",
        "bad_prior_obj1_rank",
        "bad_prior_obj2_rank",
    ]].corr(method="spearman")


    return ranked_correlations  # noqa: RET504


def sample_and_eval(
    benchmark: BenchmarkDescription,
    n=100,
    seed=42,
) -> list[Result]:
    """Sample configurations from the benchmark and evaluate them."""
    sampled_configs: list[Query] = cs_random_sampling(
        benchmark=benchmark,
        nsamples=n,
        seed=seed,
        at=first(benchmark.fidelities)[1].max,
    )
    results: list[Result] = []
    bench=benchmark.load(benchmark)
    for i, query in enumerate(sampled_configs):
        if (i+1) % 10 == 0:
            print(f"Sampled config {i+1}")
        query.config.config_id = str(i)
        result = bench.query(query)
        results.append(result)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark to generate prior ref points for",
    )
    parser.add_argument(
        "--priors",
        type=str,
        nargs="+",
        help="List of priors in the format <objective_name>:<prior_type>"
    )
    parser.add_argument(
        "--priors_dir",
        type=Path,
        default=DEFAULT_PRIORS_DIR,
        help="Directory containing prior files",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where benchmark data is stored",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to draw for rank correlation computation",
    )

    args = parser.parse_args()

    benchmark = BENCHMARKS(datadir=args.data_dir)[args.benchmark]
    neps_space = convert_configspace(benchmark.config_space)

    objectives = []

    priors = {}
    for prior in args.priors:
        objective_name, prior_type = prior.split(":")
        objectives.append(objective_name)
        prior_path = args.priors_dir / f"{args.benchmark}_{objective_name}_{prior_type}.yaml"
        with prior_path.open("r") as f:
            prior_data = yaml.safe_load(f)
            priors[prior] = prior_data["config"]
    objectives = list(set(objectives))


    priors: tuple[str, Mapping[str, Mapping[str, Any]]] = (
        f"{args.benchmark}_priors",
        priors,
    )

    prior_confidences = {
        obj_prior_type: dict.fromkeys(
            prior.keys(),
            0.75
        )
        for obj_prior_type, prior in priors[1].items()
    }
    dists = MOPriorSampler.dists_from_centers_and_confidences(
        parameters = neps_space.searchables,
        prior_centers=priors[1],
        confidence_values=prior_confidences,
    )

    encoder = ConfigEncoder.from_parameters(neps_space.searchables)


    ranked_corr_df = rank_correlation(
        benchmark=benchmark,
        prior_dists=dists,
        encoder=encoder,
        objective1=objectives[0],
        objective2=objectives[1],
        n=args.num_samples,
    )

    print("Rank Correlation Matrix:")
    print(ranked_corr_df)

    ranked_corr_df.to_csv(
        Path.cwd() / f"{args.benchmark}_rank_correlation.csv"
    )

