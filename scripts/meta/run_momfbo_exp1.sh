#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080

# Define a name for your job
#SBATCH --job-name MOMFBO_exp1            #  short: -J MOMFBO_exp1

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A_meta-momfbo.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A_meta-momfbo.out
#SBATCH --error logs/%x-%A_meta-momfbo.err    # STDERR  short: -e logs/%x-%A_meta-momfbo.out

# Since using GPU Partion, define the number of gpus required per node
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=6

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node, $SLURM_GPUS gpus with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/repos/momfp_env/bin/activate

# Running the job
start=`date +%s`

python3 -m momfpriors.run -y configs/exp_configs/exp1.yaml -e "all_25"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime