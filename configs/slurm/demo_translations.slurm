#!/bin/bash
#SBATCH --job-name=demo-trans
#SBATCH --output=/home/it238/projects/mapudungun/mapudungun-mt/logs/demo-trans_%j.out
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=matrix

export HF_HUB_OFFLINE=1

cd /home/it238/projects/mapudungun/mapudungun-mt
uv run python scripts/eval/demo_translations.py
