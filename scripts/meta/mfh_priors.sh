#!/bin/bash
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --job-name mfh_priors
#SBATCH --output logs/%x-%A_meta-prior-gen.out
#SBATCH --error logs/%x-%A_meta-prior-gen.err
#SBATCH --cpus-per-task 30

# Submit from the repository root after creating the logs directory:
# mkdir -p logs
# sbatch scripts/meta/mfh_priors.sh

set -euo pipefail

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on $SLURM_JOB_PARTITION"

source ~/repos/envs/momfp_env/bin/activate
start=$(date +%s)

# Benchmark discovery also loads the data-backed suites.
data_dir="${DATA_DIR:-/work/dlclarge2/basus-basus_ws/data/}"

# Generate good and bad priors for both objectives of all 3D/6D MFH variants.
# The existing YAML writes the generated priors to src/priors.
python3 -m momfpriors.generate_priors \
  -y configs/prior_gen/mfh_mo.yaml \
  -n 100000 \
  -s 0 \
  --data_dir "$data_dir"

end=$(date +%s)
echo "Job execution complete. Runtime: $((end - start)) seconds"
