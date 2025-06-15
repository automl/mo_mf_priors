#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name ablation1
#SBATCH --output logs/%x-%A_%a_meta.out
#SBATCH --error logs/%x-%A_%a_meta.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-8   # Still only 9 jobs
#SBATCH --time=2-00:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME, task $SLURM_ARRAY_TASK_ID"

source ~/repos/momfp_env/bin/activate

start=$(date +%s)

# === Optimizers
with_priors=(
    # "NepsPriMO"
)

without_priors=(
    "NepsMOASHA_RS"
    # "NepsMOASHABO"
    # "NepsRW"
    # "NepsPiBORW"
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

priors=("good:good" "bad:good" "bad:bad")

# === Enumerate all configs
job_list=()

# With priors: 2 opts × 9 benches × 3 priors = 54
for opt in "${with_priors[@]}"; do
    for bench_line in "${benchmarks[@]}"; do
        read -r bench key1 key2 <<< "$bench_line"
        for prior in "${priors[@]}"; do
            IFS=":" read -r val1 val2 <<< "$prior"
            job_list+=("$opt:$bench:$key1:$val1:$key2:$val2")
        done
    done
done

# Without priors: 4 opts × 9 benches = 36
for opt in "${without_priors[@]}"; do
    for bench_line in "${benchmarks[@]}"; do
        read -r bench key1 key2 <<< "$bench_line"
        job_list+=("$opt:$bench:$key1:null:$key2:null")
    done
done

# === Get current job
job="${job_list[$SLURM_ARRAY_TASK_ID]}"
IFS=":" read -r optimizer benchmark obj1 val1 obj2 val2 <<< "$job"

# === Create config
config_dir="generated_configs"
mkdir -p "$config_dir"
yaml_file="${config_dir}/ablation1_${SLURM_ARRAY_TASK_ID}.yaml"

cat > "$yaml_file" <<EOF
optimizers:
  - name: $optimizer
benchmarks:
  - name: $benchmark
    objectives:
      $obj1: $val1
      $obj2: $val2
num_seeds: 25
num_iterations: 25
EOF

echo "Generated config:"
cat "$yaml_file"

# === Run experiment
python3 -m momfpriors.run -y "$yaml_file" -e "ablation1_25"

end=$(date +%s)
runtime=$((end - start))
echo "Job complete. Runtime: $runtime seconds."
