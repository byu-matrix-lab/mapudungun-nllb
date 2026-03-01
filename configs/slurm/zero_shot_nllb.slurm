#!/bin/bash
# =============================================================================
# Zero-shot NLLB-200 baseline on Mapudungun–Spanish test set.
# No fine-tuning — evaluates the base model directly.
#
# Submit one job per direction/approach combination:
#   APPROACH=lines SRC=arn TGT=es sbatch configs/slurm/zero_shot_nllb.slurm
#   APPROACH=lines SRC=es  TGT=arn sbatch configs/slurm/zero_shot_nllb.slurm
#   APPROACH=blocks SRC=arn TGT=es sbatch configs/slurm/zero_shot_nllb.slurm
#   APPROACH=blocks SRC=es  TGT=arn sbatch configs/slurm/zero_shot_nllb.slurm
# =============================================================================

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=matrix
#SBATCH --job-name="zs-nllb-${APPROACH:-lines}-${SRC:-arn}-${TGT:-es}"
#SBATCH --output=logs/%x_%j.out

echo "========================================================="
echo "SLURM JOB ID : $SLURM_JOB_ID"
echo "NODE         : $SLURM_JOB_NODELIST"
echo "APPROACH     : ${APPROACH}"
echo "DIRECTION    : ${SRC} -> ${TGT}"
echo "========================================================="

if [[ -z "$APPROACH" || -z "$SRC" || -z "$TGT" ]]; then
    echo "ERROR: APPROACH, SRC, and TGT must be set before calling sbatch."
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
DATA_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/data-processed"
OUTPUT_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/results/zero_shot"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

srun uv run python scripts/eval/zero_shot_nllb.py \
    --model      "facebook/nllb-200-distilled-600M" \
    --approach   "${APPROACH}" \
    --src        "${SRC}" \
    --tgt        "${TGT}" \
    --data-dir   "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --max-length 128 \
    --num-beams  4
