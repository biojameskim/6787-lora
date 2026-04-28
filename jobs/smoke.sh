#!/bin/bash
#SBATCH -J peft_smoke
#SBATCH -o /share/j_sun/jjk297/repos/6787-lora/slurm/smoke_%j.out
#SBATCH -e /share/j_sun/jjk297/repos/6787-lora/slurm/smoke_%j.err
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=jjs533,gpu

set -euo pipefail

cd /share/j_sun/jjk297/repos/6787-lora
export PYTHONNOUSERSITE=1
PY=/share/j_sun/jjk297/repos/6787-lora/.venv/bin/python

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Python: $($PY --version) at $PY"
echo "transformers: $($PY -c 'import transformers; print(transformers.__version__, transformers.__file__)')"

$PY scripts/smoke_test.py "$@"
