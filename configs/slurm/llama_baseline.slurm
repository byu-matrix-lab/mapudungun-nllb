#!/bin/bash
# =============================================================================
# Llama-3.3-70B-Instruct (4-bit quant) few-shot baseline for Mapudungun–Spanish translation.
#
# Submit one job per direction (--job-name sets the log filename):
#   SRC=arn TGT=es  sbatch --job-name=llama-lines-arn-es configs/slurm/llama_baseline.slurm
#   SRC=es  TGT=arn sbatch --job-name=llama-lines-es-arn configs/slurm/llama_baseline.slurm
#
# Optionally override approach:
#   APPROACH=blocks SRC=arn TGT=es sbatch --job-name=llama-blocks-arn-es configs/slurm/llama_baseline.slurm
# =============================================================================

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --qos=matrix
#SBATCH --output=logs/%x_%j.out

echo "========================================================="
echo "SLURM JOB ID : $SLURM_JOB_ID"
echo "NODE         : $SLURM_JOB_NODELIST"
echo "APPROACH     : ${APPROACH:-lines}"
echo "DIRECTION    : ${SRC} -> ${TGT}"
echo "========================================================="

if [[ -z "$SRC" || -z "$TGT" ]]; then
    echo "ERROR: SRC and TGT must be set before calling sbatch."
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
DATA_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/data-processed"
OUTPUT_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/results/llama"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

srun uv run python scripts/eval/llama_baseline.py \
    --model        "meta-llama/Llama-3.3-70B-Instruct" \
    --approach     "${APPROACH:-lines}" \
    --src          "${SRC}" \
    --tgt          "${TGT}" \
    --data-dir     "${DATA_DIR}" \
    --output-dir   "${OUTPUT_DIR}" \
    --num-shots    5 \
    --batch-size   2 \
    --max-new-tokens 200
