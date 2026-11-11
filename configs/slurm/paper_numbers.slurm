#!/bin/bash
# v2 paper: every data-derived number in the paper (scripts/analysis/paper_numbers.py). CPU only.
#SBATCH --job-name=paper-numbers
#SBATCH --partition=m9
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/analysis/paper_numbers.py
