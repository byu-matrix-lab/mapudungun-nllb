#!/bin/bash
# Did UnigramLM / Optuna BPE / Morfessor-BPE train on segmented data? See scripts/analysis/v2_loader_check.py.
#SBATCH --job-name=v2-loader-check
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs-1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/analysis/v2_loader_check.py
