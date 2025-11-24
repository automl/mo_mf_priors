#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name naive_jahs
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-23%18   # (2 prior opts * 4 priors) * 3 benchmarks = 24 total combinations
#SBATCH --time=4-00:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/jahs_env/bin/activate

start=$(date +%s)

# Optimizers
prior_opts=(
    # "RandomSearchWithPriors"
    # "NepsMOPriorband"
    "NepsPriorMOASHA"
    # "NepsPiBORW"
    # "NepsPriMO"
    "NepsPriorRSMOASHA"
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

for opt in "${prior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    for setting in "${prior_settings[@]}"; do
      total_jobs+=("$opt:$bench:$setting")
    done
  done
done

# Print total jobs
echo "Total jobs: ${#total_jobs[@]}"

# === Pick current job
job="${total_jobs[$SLURM_ARRAY_TASK_ID]}"
IFS=":" read -r optimizer benchmark obj1 obj2 <<< "$job"

# Map keys
key1="perplexity"
key2="flops"

# === Create YAML ===
config_dir="generated_configs"
mkdir -p "$config_dir"
yaml_file="${config_dir}/${benchmark}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

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
kwargs:
  device: "rtx2080"
EOF

echo "Generated config:"
cat "$yaml_file"

data_dir="/work/dlclarge2/basus-basus_ws/data/"

# === Run the experiment ===
python3 -m momfpriors.run -y "$yaml_file" -e "naive_20" --data_dir "$data_dir"

end=$(date +%s)
runtime=$((end - start))

echo "Finished in $runtime seconds"
