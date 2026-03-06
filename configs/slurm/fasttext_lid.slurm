#!/bin/bash
#SBATCH --job-name=fasttext-lid
#SBATCH --output=logs/fasttext-lid_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --qos=standby

cd /home/it238/projects/mapudungun/mapudungun-mt
uv run python scripts/analysis/fasttext_lid_analysis.py
