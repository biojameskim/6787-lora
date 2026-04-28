#!/bin/bash
#SBATCH -J exp3_peft
#SBATCH -o /share/j_sun/jjk297/repos/6787-lora/slurm/exp3_%A_%a.out
#SBATCH -e /share/j_sun/jjk297/repos/6787-lora/slurm/exp3_%A_%a.err
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --partition=jjs533,gpu
#SBATCH --array=0-17%2

set -euo pipefail

cd /share/j_sun/jjk297/repos/6787-lora
export PYTHONNOUSERSITE=1
PY=/share/j_sun/jjk297/repos/6787-lora/.venv/bin/python

shopt -s nullglob globstar
CONFIGS=(configs/exp3/**/*.yaml)
N=${#CONFIGS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "$N" ]; then
    echo "[exit] array index ${SLURM_ARRAY_TASK_ID} >= ${N} configs"; exit 0
fi
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Job: ${SLURM_JOB_ID:-local} array ${SLURM_ARRAY_TASK_ID}/${N}  cfg=${CFG}"
$PY -m src.train --config "${CFG}"
