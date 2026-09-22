#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition dev_cpu    # short: -p single
#SBATCH --job-name debug_primo            #  short: -J debug_primo
#SBATCH --output logs/%x-%A_bwuni3.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A_bwuni3.err    # STDERR  short: -e logs/%x-%A-job_name.out
#SBATCH --time 0:30:00                      #  short: -t 0:30:00
# #SBATCH --mem 4GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

opts=(
    "NepsPriMO"
)

benchmarks=(
    "yahpo-lcbench-168330"
)


# Prior benchmark settings (good-good, bad-good, bad-bad)
prior_settings=(
  "good:good"
)

# === Compute total jobs
total_jobs=()

for opt in "${opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    for setting in "${prior_settings[@]}"; do
      total_jobs+=("$opt:$bench:$setting")
    done
  done
done

echo "Total jobs: ${#total_jobs[@]}"

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
  echo "Error: Unknown benchmark objective keys for benchmark=$benchmark"
  exit 1
fi

# === Create YAML ===
config_dir="generated_configs"
mkdir -p "$config_dir"

# Generate a debug config file
yaml_file="$config_dir/debug_config.yaml"

cat > "$yaml_file" <<EOF
optimizers:
  - name: $optimizer
benchmarks:
  - name: $benchmark
    objectives:
      $key1: ${obj1}
      $key2: ${obj2}
num_seeds: 1
num_iterations: 20
EOF

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/repos/momfp_env/bin/activate

# Running the job
start=`date +%s`

data_dir="/pfs/work9/workspace/scratch/tu_iiocv01-primo_ws"

python3 -m momfpriors.run \
-y $yaml_file \
-e "debug" \
--data_dir "$data_dir"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime