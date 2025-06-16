#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name subset1_25
#SBATCH --output logs/%x-%A_%a_meta.out
#SBATCH --error logs/%x-%A_%a_meta.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0   # (1*3*9) = 27 total combinations
#SBATCH --time=2-00:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/momfp_env/bin/activate

start=$(date +%s)

# Optimizers
prior_opts=(
  # "RandomSearchWithPriors"
  "NepsPriMO"
)
nonprior_opts=(
  # "RandomSearch"
  # "SMAC_ParEGO"
  # "NepsRW"
  # "NepsHyperbandRW"
  # "Nevergrad_EvolutionStrategy"
  # "NepsMOASHA"
)

# Benchmarks with known objective types (used for both prior and non-prior)
benchmarks=(
  # "MOMFPark"
  # "pd1-cifar100-wide_resnet-2048"
  # "pd1-imagenet-resnet-512"
  # "pd1-lm1b-transformer-2048"
  # "pd1-translate_wmt-xformer_translate-64"
  # "yahpo-lcbench-126026"
  # "yahpo-lcbench-146212"
  # "yahpo-lcbench-168330"
  "yahpo-lcbench-168868"
)

# Prior benchmark settings (good-good, bad-good, bad-bad)
prior_settings=(
  # "good:good"
  "bad:good"
  # "bad:bad"
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
if [[ "$benchmark" == MOMFPark ]]; then
  key1="value1"
  key2="value2"
elif [[ "$benchmark" == pd1-* ]]; then
  key1="valid_error_rate"
  key2="train_cost"
elif [[ "$benchmark" == yahpo-lcbench-* ]]; then
  key1="val_cross_entropy"
  key2="time"
else
  key1="UNKNOWN_KEY1"
  key2="UNKNOWN_KEY2"
fi

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
seeds: 3356806662
num_iterations: 100
EOF

echo "Generated config:"
cat "$yaml_file"

# === Run the experiment ===
python3 -m momfpriors.run -y "$yaml_file" -e "subset1_100"

end=$(date +%s)
runtime=$((end - start))

echo "Finished in $runtime seconds"
