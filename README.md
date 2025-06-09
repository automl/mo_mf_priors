# Multi-objective Multi-fidelity Optimization with Priors

# Reproducibility Guide

## Experimental Seeds

To ensure reproducible results across experiments, our code uses a deterministic seed generation function. All experiments use a global seed value of `GLOBAL_SEED = 42` to generate the required number of seeds to actually run the experiments.

### Generated Seeds (n=25)

The following 25 seeds are automatically generated for the experiments in the thesis:

| Seed Number | Seed Value | Seed Number | Seed Value | Seed Number | Seed Value |
|-------|-----------|-------|-----------|-------|-----------|
| 1     | 383329928 | 10    | 404488629 | 19    | 3606691182 |
| 2     | 3324115917| 11    | 2261209996| 20    | 1934392873 |
| 3     | 2811363265| 12    | 4190266093| 21    | 2148995113 |
| 4     | 1884968545| 13    | 3160032369| 22    | 1592565387 |
| 5     | 1859786276| 14    | 3269070127| 23    | 784044719 |
| 6     | 3687649986| 15    | 3081541440| 24    | 3980425318 |
| 7     | 369133709 | 16    | 3376120483| 25    | 3356806662 |
| 8     | 2995172878| 17    | 2204291347|       |           |
| 9     | 865305067 | 18    | 550243862 |       |           |

## Output Data Structure

Each optimization run generates a `pandas.DataFrame` containing comprehensive experimental results and metadata. The DataFrame follows a standardized schema with the following columns:

| Column | Description |
|--------|-------------|
| `budget_cost` | Optimization budget cost. For BB optimizers, this is 1 every iteration; for MF optimizers we take a *fractional* cost: `z/z_max`, where `z` is the fidelity at which a config is evaluated and `z_max` is the *maximum fidelity* available in the benchmark. |
| `budget_used_total` | Cumulative addition of `budget_cost` at every iteration. |
| `continuations_budget_cost` | Optimization budget cost in a *continual* setup, i.e., for configs at a higher fidelity (`z'`) than already evaluated (`z`), the effective cost is `(z'-z)/z_max`. |
| `continuations_budget_used_total` | Cumulative addition of `continuations_budget_cost` at every iteration. |
| `continuations_cost` | Direct fidelity cost in a *continual* optimization run: `z'-z`. |
| `fidelity` | Fidelity at which the current config is evaluated.|
| `config_id` | Unique identifier for the configuration |
| `config` | Complete configuration dictionary. |
| `results` | Dictionary containing the result values of the evaluated config. |
| `seed` | Random seed used for this specific run. |
| `budget_total` | Maximum available budget for the run. |
| `optimizer` | Name of the optimizer used. |
| `optimizer_hyperparameters` | Hyperparameter settings for the optimizer. |
| `benchmark` | Benchmark name. |
| `prior_annotations` | Any prior knowledge or annotations, e.g.: `good-good` signifies good priors for a 2-objective MO `hpoglue` `Problem`. |
| `objectives` | Optimization objectives. |
| `minimize` | Dict of booleans indicating minimization vs maximization for each objective. |
| `problem.fidelity.count` | Number of fidelity dimensions in the `Problem`. |
| `problem.fidelity.1.name` | Name of the first fidelity dimension in the `Problem`. |
| `problem.fidelity.1.min` | Minimum value for first fidelity dimension in the `Problem`. |
| `problem.fidelity.1.max` | Maximum value for first fidelity dimension in the `Problem`. |
| `benchmark.fidelity.count` | Number of benchmark fidelity dimensions. |
| `benchmark.fidelity.1.name` | Name of the first benchmark fidelity dimension. |
| `benchmark.fidelity.1.min` | Minimum value for the first fidelity dimension in the benchmark.|
| `benchmark.fidelity.1.max` | Maximum value for the first fidelity dimension in the benchmark. |

---