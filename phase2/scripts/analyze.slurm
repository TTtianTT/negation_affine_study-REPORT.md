#!/bin/bash
#SBATCH --job-name=negation-p2-analysis
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=phase2/logs/analyze-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=/tmp/negation-phase2-mpl
PY=/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python
srun "$PY" phase2/scripts/run_c.py
srun "$PY" phase2/scripts/run_ab.py
