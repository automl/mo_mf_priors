#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name bo_comp
#SBATCH --output logs/%x-%A_%a_meta.out
#SBATCH --error logs/%x-%A_%a_meta.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-35  # (1 NepsRW + 3 NepsMOASHABO variants) × 9 benchmarks = 36
#SBATCH --time=2-00:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME, task $SLURM_ARRAY_TASK_ID"

source ~/repos/momfp_env/bin/activate

start=$(date +%s)

# === Optimizers (no priors)
optimizers=(
    "NepsRW:"
    "NepsMOASHABO:5"
    "NepsMOASHABO:7"
    "NepsMOASHABO:10"
)

# === Benchmarks with keys
benchmarks=(
    "MOMFPark value1 value2"
    "pd1-cifar100-wide_resnet-2048 valid_error_rate train_cost"
    "pd1-imagenet-resnet-512 valid_error_rate train_cost"
    "pd1-lm1b-transformer-2048 valid_error_rate train_cost"
    "pd1-translate_wmt-xformer_translate-64 valid_error_rate train_cost"
    "yahpo-lcbench-126026 val_cross_entropy time"
    "yahpo-lcbench-146212 val_cross_entropy time"
    "yahpo-lcbench-168330 val_cross_entropy time"
    "yahpo-lcbench-168868 val_cross_entropy time"
)

# === Generate job list
job_list=()
for opt_line in "${optimizers[@]}"; do
    IFS=":" read -r opt init_size <<< "$opt_line"
    for bench_line in "${benchmarks[@]}"; do
        read -r bench key1 key2 <<< "$bench_line"
        job_list+=("$opt:$init_size:$bench:$key1:$key2")
    done
done

# === Select current job
job="${job_list[$SLURM_ARRAY_TASK_ID]}"
IFS=":" read -r optimizer init_size benchmark obj1 obj2 <<< "$job"

# === Create config
config_dir="generated_configs"
mkdir -p "$config_dir"
yaml_file="${config_dir}/bo_comp_${SLURM_ARRAY_TASK_ID}.yaml"

{
    echo "optimizers:"
    echo "  - name: $optimizer"
    if [[ "$optimizer" == "NepsMOASHABO" ]]; then
        echo "    hyperparameters:"
        echo "      initial_design_size: $init_size"
    fi
    echo "benchmarks:"
    echo "  - name: $benchmark"
    echo "    objectives:"
    echo "      $obj1: null"
    echo "      $obj2: null"
    echo "num_seeds: 25"
    echo "num_iterations: 25"
} > "$yaml_file"

echo "Generated config:"
cat "$yaml_file"

# === Run experiment
python3 -m momfpriors.run -y "$yaml_file" -e "bo_comp_25"

end=$(date +%s)
runtime=$((end - start))
echo "Job complete. Runtime: $runtime seconds."
