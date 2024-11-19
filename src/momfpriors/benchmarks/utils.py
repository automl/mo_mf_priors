import numpy as np
import pandas as pd
from hpoglue import BenchmarkDescription
from hpoglue import Query, Config

def find_incumbent(
    df: pd.DataFrame,
    results_col: str,
    objective: str,
    minimize: bool = True,
) -> float:
    """
    Find the best value of the objective in the results column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the results.
    results_col : str
        The name of the column containing the results.
    objective : str
        The name of the objective to minimize.

    Returns
    -------
    float
        The best value of the objective.
    """
    if df.empty:
        return np.nan
    vals = []
    for i in df[results_col]:
        vals.append(i[objective])
    if minimize:
        return min(vals)
    else:
        return max(vals)
    

def bench_first_fid(benchmark: BenchmarkDescription) -> int:
    bench_first_fid = benchmark.fidelities[
        list(benchmark.fidelities.keys())[0]
    ]
    return bench_first_fid


def cs_random_sampling(
    benchmark: BenchmarkDescription,
    nsamples: int,
    seed: int,
    at: int,
    ) -> list[Query]:
    """
    Sample configurations from the benchmark's configuration space.
    
    Args:
        benchmark (BenchmarkDescription): The benchmark description.

        nsamples (int): The number of samples to draw from the configuration space.

        seed (int): The seed for random number generation.

    Returns:
        list[Result]: The sampled configurations.
    """
    benchmark.config_space.seed(seed)
    configs = benchmark.config_space.sample_configuration(nsamples)
    queries = [
        Query(
            config=Config(
                config_id=None,
                values=dict(config),
            ), 
            fidelity=(bench_first_fid(benchmark), at)
        ) 
            for config in configs
    ]

    return queries

