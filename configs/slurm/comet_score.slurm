#!/bin/bash
# =============================================================================
# Score all prediction files with COMET (wmt22-comet-da).
# Uses the shared checkpoint from grp_ladle to avoid re-downloading.
#
# Submit:
#   sbatch --job-name=comet-score configs/slurm/comet_score.slurm
# =============================================================================

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=matrix
#SBATCH --output=logs/%x_%j.out

echo "========================================================="
echo "SLURM JOB ID : $SLURM_JOB_ID"
echo "NODE         : $SLURM_JOB_NODELIST"
echo "========================================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
RESULTS_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/predictions"
COMET_CKPT="/home/$(whoami)/groups/grp_ladle/nobackup/autodelete/comet_model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"

cd "${PROJECT_DIR}"
mkdir -p logs

srun uv run python scripts/eval/comet_score.py \
    --results-dir  "${RESULTS_DIR}" \
    --comet-model  "${COMET_CKPT}" \
    --batch-size   64 \
    --gpus         1
