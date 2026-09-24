#!/bin/bash
#SBATCH --partition cpu
#SBATCH --job-name all_100
#SBATCH --output logs/%x-%A_%a_bwuni3.out
#SBATCH --error logs/%x-%A_%a_bwuni3.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-5   # (1 prior opts * 4 priors + 2 non-prior opts) * 1 benchmarks = 6 total combinations
#SBATCH --time=3-00:00:00
# #SBATCH --mem 30G

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/envs/momfp_env/bin/activate

start=$(date +%s)

# Optimizers
prior_opts=(
  "NepsPriMO"
)
nonprior_opts=(
  # "RandomSearch"
  # "SMAC_ParEGO"
  "NepsRW"
  # "NepsHyperbandRW"
  # "Nevergrad_EvolutionStrategy"
  # "NepsMOASHA"
  "Optuna"
  # "NepsMOBO"
)

# Benchmarks with known objective types (used for both prior and non-prior)
benchmarks=(
  "pd1-translate_wmt-xformer_translate-64"
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

# 1 prior opts × 4 priors
for opt in "${prior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    for setting in "${prior_settings[@]}"; do
      total_jobs+=("$opt:$bench:$setting")
    done
  done
done

# 2 non-prior opts × 1 benchmarks (with nulls)
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
if [[ "$benchmark" == pd1-* ]]; then
  key1="valid_error_rate"
  key2="train_cost"
else
  key1="UNKNOWN_KEY1"
  key2="UNKNOWN_KEY2"
fi

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
num_iterations: 100
EOF

echo "Generated config:"
cat "$yaml_file"


data_dir="/pfs/work9/workspace/scratch/tu_iiocv01-primo_ws"

# === Run the experiment ===
python3 -m momfpriors.run \
-y "$yaml_file" \
-e "all_100_evals" \
--data_dir "$data_dir"

end=$(date +%s)
runtime=$((end - start))

echo "Finished in $runtime seconds"
