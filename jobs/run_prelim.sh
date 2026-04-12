#!/bin/bash
#SBATCH -J prelim
#SBATCH -o /share/j_sun/jjk297/repos/6787-lora/slurm/prelim_%j.out
#SBATCH -e /share/j_sun/jjk297/repos/6787-lora/slurm/prelim_%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --partition=jjs533
#SBATCH --nodelist=jjs533-compute-03

# Set up conda environment
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Navigate to the working directory
cd /share/j_sun/jjk297/repos/6787-lora

source 6787-lora/bin/activate

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

python preliminary_exploration.py
