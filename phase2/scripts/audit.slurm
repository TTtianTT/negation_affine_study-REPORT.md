#!/bin/bash
#SBATCH --job-name=negation-p2-audit
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=phase2/logs/audit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
srun /dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python phase2/scripts/validate.py
