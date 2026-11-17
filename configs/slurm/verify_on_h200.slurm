#!/bin/bash
# Repeat the reconstruction check for the two mismatching models on an H200, the GPU
# type their original predictions were generated on (eng-1-1; request standby, which the cluster maps to gstandby).
#SBATCH --job-name=MA_verifyH200
#SBATCH --partition=eng
#SBATCH --qos=standby
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python -u scripts/release/verify_on_h200.py
