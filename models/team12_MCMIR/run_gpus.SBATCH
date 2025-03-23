#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:rtx8000:1
#sbatch --account=pr_107_general
#SBATCH --job-name=mambaIR_test
#SBATCH --output=mambaIR_test%j.txt
#SBATCH --error=mambaIR_test%j.err
module purge
# Define paths
SINGULARITY_IMAGE="/scratch/ll5484/dinov2/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif"
CONDA_ENV_PATH="/scratch/ll5484/conda_envs/mambaIR"
SCRIPT_PATH="basicsr/train.py"

singularity exec --nv \
    --overlay /scratch/ll5484/dinov2/overlay-50G-10M.ext3:ro \
    $SINGULARITY_IMAGE \
    /bin/bash -c " \
    source /ext3/env.sh && \
    conda activate $CONDA_ENV_PATH && \
    CUDA_LAUNCH_BLOCKING=1 python $SCRIPT_PATH"
# --data_dir /scratch/ll5484/ntire25/dataset/LSDIR \
