#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name subset1_100
#SBATCH --output logs/%x-%A_%a_meta.out
#SBATCH --error logs/%x-%A_%a_meta.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-107%20   # 108 total combinations

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME, task $SLURM_ARRAY_TASK_ID";

source ~/repos/momfp_env/bin/activate

# === Config generation ===
config_dir="generated_configs"
mkdir -p "$config_dir"

if [[ ! -f "${config_dir}/.generated" ]]; then
    echo "Generating YAML configs in Bash..."

    optimizers=(
        "RandomSearchWithPriors"
        "RandomSearch"
        "SMAC_ParEGO"
        "NepsRW"
        "NepsHyperbandRW"
        "Nevergrad_EvolutionStrategy"
        "NepsMOASHA"
        "NepsMOASHAPiBORW"
    )

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

    priors=("good good" "bad good" "bad bad")

    config_id=0

    for optimizer in "${optimizers[@]}"; do
        for benchmark_line in "${benchmarks[@]}"; do
            read -r name obj1 obj2 <<< "$benchmark_line"
            for prior_pair in "${priors[@]}"; do
                read -r val1 val2 <<< "$prior_pair"

                yaml_file="${config_dir}/prior_exp_${config_id}.yaml"
                {
                    echo "optimizers:"
                    echo "  - name: $optimizer"
                    echo "benchmarks:"
                    echo "  - name: $name"
                    echo "    objectives:"
                    echo "      $obj1: $val1"
                    echo "      $obj2: $val2"
                    echo "num_seeds: 25"
                    echo "num_iterations: 100"
                } > "$yaml_file"

                ((config_id++))
            done
        done
    done

    touch "${config_dir}/.generated"
    echo "YAML generation complete: $config_id files."
else
    echo "Configs already exist. Skipping generation."
fi

# === Run job for this task ID ===
yaml_file="${config_dir}/prior_exp_${SLURM_ARRAY_TASK_ID}.yaml"

if [[ ! -f "$yaml_file" ]]; then
    echo "ERROR: Config file $yaml_file does not exist!"
    exit 1
fi

echo "Running with config: $yaml_file"
cat "$yaml_file"

start=$(date +%s)

python3 -m momfpriors.run -y "$yaml_file" -e "subset1_100"

end=$(date +%s)
runtime=$((end - start))

echo "Job complete. Runtime: $runtime seconds."
