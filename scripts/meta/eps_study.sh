#!/bin/bash
#SBATCH --partition mldlc2_cpu-epyc9655
#SBATCH --job-name primo_eps_study
#SBATCH --output logs/%x-%A_%a_meta.out
#SBATCH --error logs/%x-%A_%a_meta.err
#SBATCH --cpus-per-task 1
#SBATCH --array=0-139%20 # (1 prior opts * 4 priors * 5 hp settings) * 7 benchmarks = 140 total combinations
# #SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu 30G

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/momfp_env/bin/activate

start=$(date +%s)

prior_opts=(
  "NepsPriMO"
)

benchmarks=(
  "pd1-imagenet-resnet-512"
  "pd1-cifar100-wide_resnet-2048"
  "pd1-lm1b-transformer-2048"
  # "pd1-translate_wmt-xformer_translate-64"
  "yahpo-lcbench-126026"
  "yahpo-lcbench-146212"
  "yahpo-lcbench-168330"
  "yahpo-lcbench-168868"
)

prior_settings=(
  "good:good"
  "bad:good"
  "bad:bad"
  "good:bad"
)

hp="epsilon"

hp_settings=(
  "0.1"
  "0.3"
  "0.5"
  "0.7"
  "0.9"
)

total_jobs=()

for opt in "${prior_opts[@]}"; do
  for eps in "${hp_settings[@]}"; do
    for obj_setting in "${prior_settings[@]}"; do
      for bench in "${benchmarks[@]}"; do
        total_jobs+=("${opt}:${eps}:${obj_setting}:${bench}")
      done
    done
  done
done

echo "Total jobs: ${#total_jobs[@]}"

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#total_jobs[@]}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds total jobs=${#total_jobs[@]}"
  exit 1
fi

job="${total_jobs[$SLURM_ARRAY_TASK_ID]}"

IFS=":" read -r optimizer epsilon obj1 obj2 benchmark <<< "$job"

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

echo "Optimizer: $optimizer"
echo "Benchmark: $benchmark"
echo "Epsilon: $epsilon"
echo "Objectives: $key1=$obj1, $key2=$obj2"

config_dir="generated_configs"
mkdir -p "$config_dir"

yaml_file="${config_dir}/${benchmark}_${optimizer}_${hp}_${epsilon}_${obj1}_${obj2}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

cat > "$yaml_file" <<EOF
optimizers:
  - name: $optimizer
    hyperparameters:
      $hp: $epsilon
benchmarks:
  - name: $benchmark
    objectives:
      $key1: $obj1
      $key2: $obj2
num_seeds: 25
num_iterations: 20
EOF

echo "Generated config:"
cat "$yaml_file"

data_dir="/work/dlclarge2/basus-basus_ws/data/"

python3 -m momfpriors.run \
  -y "$yaml_file" \
  -e "eps_study" \
  --data_dir "$data_dir"

end=$(date +%s)
runtime=$((end - start))

echo "Finished in $runtime seconds"