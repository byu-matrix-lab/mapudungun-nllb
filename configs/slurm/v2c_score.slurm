#!/bin/bash
# v2 scoring + paired bootstrap (CPU only). Writes results/paper_stats.json["v2"].
#SBATCH --job-name=v2c-score
#SBATCH --partition=m9
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/analysis/v2c_score.py
