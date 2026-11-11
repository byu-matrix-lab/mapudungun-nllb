#!/bin/bash
# Reconstruct the incomplete paper-era 3.3B es->arn UnigramLM checkpoint.

#SBATCH --job-name=MA_release33B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Matrix A100s are often occupied by the lab's other arrays; permit the CS
# A100/H100/B200 pools for this reproducibility repair.
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --account=ekr8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

PROJECT_DIR="/home/${USER}/projects/mapudungun/mapudungun-mt"
DATA_DIR="/nobackup/archive/usr/${USER}/mapudungun_rebuild/data-processed"
OUTPUT_DIR="/nobackup/archive/usr/${USER}/mapudungun_release_models/nllb-200-3.3B-blocks-es-arn-unigram_lm"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
"$PROJECT_DIR/.venv/bin/python" scripts/training/finetune_nllb.py \
  --model facebook/nllb-200-3.3B \
  --approach blocks \
  --src es \
  --tgt arn \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer-approach unigram_lm \
  --epochs 10 \
  --batch-size 8 \
  --grad-accum 16 \
  --lr 3e-5 \
  --warmup-steps 1000 \
  --max-length 128 \
  --early-stopping-patience 3 \
  --num-beams 4 \
  --seed 42
