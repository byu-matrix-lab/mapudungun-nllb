#!/bin/bash
#SBATCH --job-name=build-human-eval
#SBATCH --output=logs/build_human_eval_%j.out
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

cd /home/it238/projects/mapudungun/mapudungun-mt
uv run python scripts/eval/build_human_eval_tsv.py
