#!/bin/bash
# Compare rebuilt checkpoints' test predictions with the lost originals' archived predictions.
#   sbatch --partition=dw --qos=matrix configs/slurm/verify_reconstructions.slurm   (or cs/cs)
#SBATCH --job-name=MA_verify
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs-1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/release/verify_reconstructions.py
