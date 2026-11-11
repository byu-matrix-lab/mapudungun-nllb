#!/bin/bash
# v2 paper: segmentation stats from the original data and models (scripts/analysis/v2_segmentation_stats.py). CPU only.
#SBATCH --job-name=v2-segstats
#SBATCH --partition=m9
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/analysis/v2_segmentation_stats.py
