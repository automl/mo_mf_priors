#!/bin/bash
#SBATCH --job-name=primo_neps_parallel_n=4
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mem-per-cpu=30000M
#SBATCH --array=0-15

source ~/repos/momfp_env/bin/activate
echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"
srun --exclusive python -m momfpriors.run_neps --exp_config neps_parallel_gen_configs/primo_neps_parallel_${SLURM_ARRAY_TASK_ID}_config.yaml --num_workers 4