#!/bin/bash
# =============================================================================
# Fine-tune NLLB-200 on Mapudungun–Spanish
# Targets the dwmatrix partition (A100) via --qos=matrix.
#
# Submit one job per direction/approach combination (--job-name sets the log filename):
#   APPROACH=lines  SRC=arn TGT=es  sbatch --job-name=nllb-lines-arn-es  configs/slurm/finetune_nllb.slurm
#   APPROACH=lines  SRC=es  TGT=arn sbatch --job-name=nllb-lines-es-arn  configs/slurm/finetune_nllb.slurm
#   APPROACH=blocks SRC=arn TGT=es  sbatch --job-name=nllb-blocks-arn-es configs/slurm/finetune_nllb.slurm
#   APPROACH=blocks SRC=es  TGT=arn sbatch --job-name=nllb-blocks-es-arn configs/slurm/finetune_nllb.slurm
# =============================================================================

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=matrix
#SBATCH --output=logs/%x_%j.out

echo "========================================================="
echo "SLURM JOB ID : $SLURM_JOB_ID"
echo "JOB NAME     : $SLURM_JOB_NAME"
echo "NODE         : $SLURM_JOB_NODELIST"
echo "GPUs         : $SLURM_GPUS_ON_NODE"
echo "APPROACH     : ${APPROACH}"
echo "DIRECTION    : ${SRC} -> ${TGT}"
echo "========================================================="

if [[ -z "$APPROACH" || -z "$SRC" || -z "$TGT" ]]; then
    echo "ERROR: APPROACH, SRC, and TGT must be set before calling sbatch."
    echo "Example: APPROACH=lines SRC=arn TGT=es sbatch configs/slurm/finetune_nllb.slurm"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
DATA_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/data-processed"
OUTPUT_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/models/nllb-600M-${APPROACH}-${SRC}-${TGT}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

# A100 (dwmatrix): batch 32 × grad_accum 4 = effective batch 128
srun uv run python scripts/training/finetune_nllb.py \
    --model      "facebook/nllb-200-distilled-600M" \
    --approach   "${APPROACH}" \
    --src        "${SRC}" \
    --tgt        "${TGT}" \
    --data-dir   "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs     10 \
    --batch-size 32 \
    --grad-accum 4 \
    --lr         3e-5 \
    --warmup-steps 1000 \
    --max-length 128 \
    --early-stopping-patience 3 \
    --num-beams  4
