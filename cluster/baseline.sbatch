#!/usr/bin/env bash
#SBATCH --job-name=loopwatch-baseline
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --constraint=node1600|node1602|node1603|node1608
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the initialized loopwatch checkout}"
# Deliberately no GPU resource request or --nv flag.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$PWD/.cache/runtime/bin:$PATH"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python -m runner.run --config configs/baseline.yaml --prepare-if-needed \
        --experiment-dir "runs/baseline-${SLURM_ARRAY_JOB_ID}" --episode-index "$SLURM_ARRAY_TASK_ID"
else
    python -m runner.run --config configs/baseline.yaml --prepare-if-needed \
        --experiment-dir "runs/baseline-${SLURM_JOB_ID}"
fi
