#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name primo_jahsbench
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-17%18   # (1 prior opts * 4 priors + 2 non-prior opts) * 3 benchmarks = 18 total combinations
#SBATCH --time=4-00:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/jahs_env/bin/activate

start=$(date +%s)

# Optimizers
prior_opts=(
  # "NepsPriMO"
  "NepsPiBORW"
  # "NepsMOPriorband"
  # "RandomSearchWithPriors"
)
nonprior_opts=(
  # "RandomSearch"
  # "SMAC_ParEGO"
  "NepsRW"
  # "NepsMOBO"
  # "NepsHyperbandRW"
  # "Nevergrad_EvolutionStrategy"
  # "NepsMOASHA"
  "Optuna"
)

# Benchmarks with known objective types (used for both prior and non-prior)
benchmarks=(
  "jahs-CIFAR10"
  "jahs-ColorectalHistology"
  "jahs-FashionMNIST"
)

# Prior benchmark settings (good-good, bad-good, bad-bad)
prior_settings=(
  "good:good"
  "bad:good"
  "bad:bad"
  "good:bad"
)

# === Compute total jobs
total_jobs=()

# 2 priors opts × 27 benchmarks (3 obj settings × 9)
for opt in "${prior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    for setting in "${prior_settings[@]}"; do
      total_jobs+=("$opt:$bench:$setting")
    done
  done
done

# 6 non-prior opts × 9 benchmarks (with nulls)
for opt in "${nonprior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    total_jobs+=("$opt:$bench:null:null")
  done
done

# Print total jobs
echo "Total jobs: ${#total_jobs[@]}"

# === Pick current job
job="${total_jobs[$SLURM_ARRAY_TASK_ID]}"
IFS=":" read -r optimizer benchmark obj1 obj2 <<< "$job"

# Map keys
key1="valid_acc"
key2="runtime"

# === Create YAML ===
config_dir="generated_configs"
mkdir -p "$config_dir"
yaml_file="${config_dir}/subset1_${SLURM_ARRAY_TASK_ID}.yaml"

cat > "$yaml_file" <<EOF
optimizers:
  - name: $optimizer
benchmarks:
  - name: $benchmark
    objectives:
      $key1: ${obj1}
      $key2: ${obj2}
num_seeds: 25
num_iterations: 20
EOF

echo "Generated config:"
cat "$yaml_file"


data_dir="/work/dlclarge2/basus-basus_ws/data/"

# === Run the experiment ===
python3 -m momfpriors.run -y "$yaml_file" -e "jahs_20_evals" --data_dir "$data_dir"

end=$(date +%s)
runtime=$((end - start))

echo "Finished in $runtime seconds"
