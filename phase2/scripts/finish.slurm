#!/bin/bash
#SBATCH --job-name=negation-p2-finish
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=phase2/logs/finish-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=/tmp/negation-phase2-mpl
PY=/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python
srun "$PY" phase2/scripts/matched.py
srun "$PY" phase2/scripts/supplement.py
srun "$PY" phase2/scripts/diagnostics.py
srun "$PY" phase2/scripts/summarize.py

srun "$PY" phase2/scripts/presentation.py
srun "$PY" phase2/scripts/report.py
srun "$PY" phase2/scripts/validate.py
