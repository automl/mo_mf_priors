#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080
#SBATCH --job-name MOMFBO_all_array
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --array=0-8   # 9 benchmarks = 9 array jobs

echo "Workingdir: $PWD";
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME, task $SLURM_ARRAY_TASK_ID";

source ~/repos/momfp_env/bin/activate

start=$(date +%s)

# === Define benchmarks ===
optimizer="MOMFBO"
benchmarks=(
    "MOMFPark"
    "pd1-cifar100-wide_resnet-2048"
    "pd1-imagenet-resnet-512"
    "pd1-lm1b-transformer-2048"
    "pd1-translate_wmt-xformer_translate-64"
    "yahpo-lcbench-126026"
    "yahpo-lcbench-146212"
    "yahpo-lcbench-168330"
    "yahpo-lcbench-168868"
)

benchmark_name=${benchmarks[$SLURM_ARRAY_TASK_ID]}

echo "Selected benchmark: $benchmark_name";

# === Create mini YAML file ===

config_dir="generated_configs"
mkdir -p $config_dir

yaml_file="${config_dir}/exp1_${SLURM_ARRAY_TASK_ID}.yaml"

cat <<EOF > "$yaml_file"
optimizers:
  - name: $optimizer
benchmarks:
  - name: $benchmark_name
    objectives:
$(if [[ "$benchmark_name" == MOMFPark ]]; then
  echo "      value1: null"
  echo "      value2: null"
elif [[ "$benchmark_name" == pd1-* ]]; then
  echo "      valid_error_rate: null"
  echo "      train_cost: null"
elif [[ "$benchmark_name" == yahpo-lcbench-* ]]; then
  echo "      val_cross_entropy: null"
  echo "      time: null"
else
  echo "      UNKNOWN_OBJECTIVE: null"
fi)
num_seeds: 25
num_iterations: 25
EOF

echo "Generated config:"
cat "$yaml_file"

# === Now run your script ===
python3 -m momfpriors.run -y "$yaml_file" -e "all_25"

end=$(date +%s)
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime seconds."
