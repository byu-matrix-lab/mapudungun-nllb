#!/bin/bash
# Reconstruct the four paper-era 600M checkpoints that were lost before release.
# Submit with: sbatch configs/slurm/rebuild_missing_600m.slurm

#SBATCH --job-name=MA_release600
#SBATCH --array=0-3
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
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

PROJECT_DIR="/home/${USER}/projects/mapudungun/mapudungun-mt"
DATA_DIR="/nobackup/archive/usr/${USER}/mapudungun_rebuild/data-processed"
OUTPUT_ROOT="/nobackup/archive/usr/${USER}/mapudungun_release_models"
CONDITIONS=(morfessor_bpe morfessor_bpe unigram_lm unigram_lm)
SOURCES=(arn es arn es)

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
SRC="${SOURCES[$SLURM_ARRAY_TASK_ID]}"
if [ "$SRC" = arn ]; then TGT=es; else TGT=arn; fi
OUTPUT_DIR="${OUTPUT_ROOT}/nllb-600M-blocks-${SRC}-${TGT}-${CONDITION}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
"$PROJECT_DIR/.venv/bin/python" scripts/training/finetune_nllb.py \
  --model facebook/nllb-200-distilled-600M \
  --approach blocks \
  --src "$SRC" \
  --tgt "$TGT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer-approach "$CONDITION" \
  --epochs 10 \
  --batch-size 32 \
  --grad-accum 4 \
  --lr 3e-5 \
  --warmup-steps 1000 \
  --max-length 128 \
  --early-stopping-patience 3 \
  --num-beams 4 \
  --seed 42
