#!/bin/bash
#SBATCH --partition cpu
#SBATCH --job-name mfh_mo_20
#SBATCH --output logs/%x-%A_%a_bwuni3.out
#SBATCH --error logs/%x-%A_%a_bwuni3.err
#SBATCH --cpus-per-task 30
#SBATCH --array=0-43%22 # (2 prior optimizers * 4 settings + 3 non-prior optimizers) * 4 benchmarks
#SBATCH --time=3-00:00:00

# Run from the repository root. Generate priors once in the activated environment:
# python3 -m momfpriors.generate_priors -y configs/prior_gen/mfh_mo.yaml -n 100000
# mkdir -p logs
# sbatch scripts/bwunicluster/mfh_mo_array.sh

set -euo pipefail

echo "Workingdir: $PWD"
echo "Started at $(date)"

source ~/repos/envs/momfp_env/bin/activate
start=$(date +%s)

prior_opts=(
  "NepsPriMO"    # PriMO
  "NepsPiBORW"   # PiBORW
)
nonprior_opts=(
  "Optuna"       # MO-TPE
  "NepsRW"       # BORW
  "NepsMOASHA"   # MO-ASHA
)

benchmarks=(
  "mfh3_bad_mo"
  "mfh3_good_mo"
  "mfh3_moderate_mo"
  "mfh3_terrible_mo"
)
prior_settings=(
  "good:good"
  "bad:good"
  "bad:bad"
  "good:bad"
)

total_jobs=()
for opt in "${prior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    for setting in "${prior_settings[@]}"; do
      total_jobs+=("$opt:$bench:$setting")
    done
  done
done
for opt in "${nonprior_opts[@]}"; do
  for bench in "${benchmarks[@]}"; do
    total_jobs+=("$opt:$bench:null:null")
  done
done

echo "Total jobs: ${#total_jobs[@]}"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array job}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id >= ${#total_jobs[@]} )); then
  echo "Invalid array task ID: $task_id (expected 0-$(( ${#total_jobs[@]} - 1 )))" >&2
  exit 1
fi
IFS=":" read -r optimizer benchmark obj1 obj2 <<< "${total_jobs[$task_id]}"

priors_dir="${PRIORS_DIR:-$PWD/src/priors}"
if [[ "$obj1" != null ]]; then
  for prior_file in "${benchmark}_value1_${obj1}.yaml" "${benchmark}_value2_${obj2}.yaml"; do
    if [[ ! -f "$priors_dir/$prior_file" ]]; then
      echo "Missing prior: $priors_dir/$prior_file" >&2
      echo "Generate priors with configs/prior_gen/mfh_mo.yaml before submitting." >&2
      exit 1
    fi
  done
fi

config_dir="generated_configs/mfh_mo"
mkdir -p "$config_dir"
yaml_file="${config_dir}/${benchmark}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${task_id}.yaml"
cat > "$yaml_file" <<EOF
optimizers:
  - name: $optimizer
benchmarks:
  - name: $benchmark
    objectives:
      value1: $obj1
      value2: $obj2
num_seeds: 25
num_iterations: 20
EOF

echo "Generated config:"
cat "$yaml_file"

data_dir="${DATA_DIR:-/pfs/work9/workspace/scratch/tu_iiocv01-primo_ws}"
python3 -m momfpriors.run \
  -y "$yaml_file" \
  -e "mfh_mo_20_evals" \
  --priors_dir "$priors_dir" \
  --data_dir "$data_dir"

end=$(date +%s)
echo "Finished in $((end - start)) seconds"
