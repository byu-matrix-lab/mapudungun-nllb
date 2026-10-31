#!/bin/bash
# =============================================================================
# v2 correction: retrain the arn->es Standard BPE baselines (notes/v2_brief.md).
# Same recipe as the paper (finetune_nllb.slurm), with these differences:
#   - training data: regenerated set under /nobackup/archive (originals were autodeleted)
#   - runs the repo .venv directly (transformers pinned 4.57.6), never `uv run`
#   - cs partition A100, cs-1-2 excluded; outputs under /nobackup/archive
#
#   MODEL=facebook/nllb-200-distilled-600M sbatch --time=6:00:00  --job-name=v2-600M-arn-es  configs/slurm/finetune_nllb_v2.slurm
#   MODEL=facebook/nllb-200-distilled-1.3B sbatch --time=8:00:00  --job-name=v2-1.3B-arn-es  configs/slurm/finetune_nllb_v2.slurm
#   MODEL=facebook/nllb-200-3.3B BATCH_SIZE=8 GRAD_ACCUM=16 \
#                                          sbatch --time=14:00:00 --job-name=v2-3.3B-arn-es  configs/slurm/finetune_nllb_v2.slurm
#
# Option (c), 256 tokens: add MAX_LENGTH=256 and TOK=standard|morfessor_vc; output gets -<tok>-len256.
# =============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs-1-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
V2_DIR="/nobackup/archive/usr/$(whoami)/mapudungun_v2"
DATA_DIR="${V2_DIR}/data"
MODEL="${MODEL:?set MODEL}"
SRC="${SRC:-arn}"
TGT="${TGT:-es}"
MODEL_TAG=$(echo "$MODEL" | sed 's|.*/||; s/nllb-200-distilled-/nllb-/')
OUTPUT_DIR="${V2_DIR}/models/${MODEL_TAG}-blocks-${SRC}-${TGT}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TOK="${TOK:-standard}"
if [ "$MAX_LENGTH" != "128" ] || [ "$TOK" != "standard" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-${TOK}-len${MAX_LENGTH}"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
PY="${PROJECT_DIR}/.venv/bin/python"

echo "JOB $SLURM_JOB_ID on $SLURM_JOB_NODELIST | $MODEL $SRC->$TGT | batch ${BATCH_SIZE}x${GRAD_ACCUM} | tok $TOK | maxlen $MAX_LENGTH | out $OUTPUT_DIR"
"$PY" -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

"$PY" scripts/training/finetune_nllb.py \
    --model               "${MODEL}" \
    --approach            blocks \
    --src                 "${SRC}" \
    --tgt                 "${TGT}" \
    --data-dir            "${DATA_DIR}" \
    --output-dir          "${OUTPUT_DIR}" \
    --tokenizer-approach  "${TOK}" \
    --epochs     10 \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr         3e-5 \
    --warmup-steps 1000 \
    --max-length "${MAX_LENGTH}" \
    --early-stopping-patience 3 \
    --num-beams  4 \
    --seed 42
