#!/bin/bash
# =============================================================================
# Optuna BPE trial: tokenize with a given vocab size, fine-tune 600M arn→es,
# evaluate on dev set, write result JSON for the coordinator.
#
# Required env vars (set via --export):
#   TRIAL_ID    Optuna trial number (used for result filename)
#   VOCAB_SIZE  BPE vocabulary size to evaluate
#
# Submit via optuna_bpe_coordinator.py (do not submit manually).
# =============================================================================

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=matrix
#SBATCH --output=logs/optuna_bpe_trial_%j.out

echo "========================================================="
echo "SLURM JOB ID : $SLURM_JOB_ID"
echo "TRIAL_ID     : ${TRIAL_ID}"
echo "VOCAB_SIZE   : ${VOCAB_SIZE}"
echo "========================================================="

if [[ -z "$TRIAL_ID" || -z "$VOCAB_SIZE" ]]; then
    echo "ERROR: TRIAL_ID and VOCAB_SIZE must be set."
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJECT_DIR="/home/$(whoami)/projects/mapudungun/mapudungun-mt"
DATA_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/data-processed"
export RESULT_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/optuna_bpe_results"
export OUTPUT_DIR="/home/$(whoami)/nobackup/autodelete/mapudungun/optuna_bpe_models/trial_${TRIAL_ID}"

cd "${PROJECT_DIR}"
mkdir -p logs "${RESULT_DIR}" "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Tokenize arn side with this vocab size → blocks/optuna_bpe/
# ---------------------------------------------------------------------------
echo "--- Tokenizing (vocab_size=${VOCAB_SIZE}) ---"
srun --ntasks=1 --gpus=0 uv run python scripts/tokenization/segment_bpe.py \
    --mode large \
    --large-vocab "${VOCAB_SIZE}" \
    --out-dir-name optuna_bpe \
    --approaches blocks

# ---------------------------------------------------------------------------
# 2. Fine-tune 600M arn→es (5 epochs, early stopping patience 3)
# ---------------------------------------------------------------------------
echo "--- Fine-tuning 600M arn→es ---"
srun uv run python scripts/training/finetune_nllb.py \
    --model               "facebook/nllb-200-distilled-600M" \
    --approach            blocks \
    --src                 arn \
    --tgt                 es \
    --data-dir            "${DATA_DIR}" \
    --output-dir          "${OUTPUT_DIR}" \
    --tokenizer-approach  optuna_bpe \
    --epochs     5 \
    --batch-size 32 \
    --grad-accum 4 \
    --lr         3e-5 \
    --warmup-steps 500 \
    --max-length 128 \
    --early-stopping-patience 3 \
    --num-beams  4

# ---------------------------------------------------------------------------
# 3. Extract best dev chrF++ from trainer_state.json and write result
# ---------------------------------------------------------------------------
echo "--- Extracting result ---"
uv run python - <<'PYEOF'
import json, glob, os, sys

trial_id   = int(os.environ["TRIAL_ID"])
vocab_size = int(os.environ["VOCAB_SIZE"])
output_dir = os.environ["OUTPUT_DIR"]
result_dir = os.environ["RESULT_DIR"]

best_metric = None
for ck in sorted(glob.glob(f"{output_dir}/checkpoint-*")):
    ts = os.path.join(ck, "trainer_state.json")
    if os.path.exists(ts):
        d = json.load(open(ts))
        m = d.get("best_metric")
        if m is not None:
            if best_metric is None or m > best_metric:
                best_metric = m

if best_metric is None:
    print("ERROR: could not find best_metric in any checkpoint", file=sys.stderr)
    sys.exit(1)

result = {"trial_id": trial_id, "vocab_size": vocab_size, "chrf": best_metric}
out_path = os.path.join(result_dir, f"trial_{trial_id}.json")
json.dump(result, open(out_path, "w"))
print(f"Trial {trial_id}: vocab_size={vocab_size} → best dev chrF++={best_metric:.4f}")
print(f"Result written to {out_path}")
PYEOF
