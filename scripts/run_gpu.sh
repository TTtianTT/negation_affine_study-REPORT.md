#!/bin/bash
set -euo pipefail
cd /dataset1/zailong/workspace/negation_affine_study
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/negation-mpl
/dataset1/zailong/workspace/outlier_text_pilot/.venv/bin/python scripts/extract.py
