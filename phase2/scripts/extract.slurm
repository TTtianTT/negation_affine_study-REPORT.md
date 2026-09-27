#!/bin/bash
#SBATCH --job-name=negation-p2-extract
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=phase2/logs/extract-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=/tmp/negation-phase2-mpl
srun /dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python phase2/scripts/extract.py
