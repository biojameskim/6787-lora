#!/bin/bash
#SBATCH -J peft_run
#SBATCH -o /share/j_sun/jjk297/repos/6787-lora/slurm/run_%j.out
#SBATCH -e /share/j_sun/jjk297/repos/6787-lora/slurm/run_%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --partition=jjs533,gpu

set -euo pipefail

CONFIG_PATH="${1:?usage: run_one.sh <path-to-config.yaml>}"

cd /share/j_sun/jjk297/repos/6787-lora
export PYTHONNOUSERSITE=1
PY=/share/j_sun/jjk297/repos/6787-lora/.venv/bin/python

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"

$PY -m src.train --config "${CONFIG_PATH}"
