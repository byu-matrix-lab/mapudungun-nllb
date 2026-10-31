#!/bin/bash
# v2: COMET (wmt22-comet-da) on mapudungun_v2/predictions. comet_score.py skips files already scored.
# The locked .venv has setuptools 82 (no pkg_resources), which torchmetrics 0.10.3 needs;
# a setuptools<81 shim is put on PYTHONPATH for this job only. The .venv itself is unchanged.
#SBATCH --job-name=v2-comet
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs-1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=/nobackup/archive/usr/it238/mapudungun_v2/comet_shim
cd /home/it238/projects/mapudungun/mapudungun-mt
.venv/bin/python scripts/eval/comet_score.py \
    --results-dir /nobackup/archive/usr/it238/mapudungun_v2/predictions \
    --comet-model /home/it238/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt \
    --batch-size 64 --gpus 1
