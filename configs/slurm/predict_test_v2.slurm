#!/bin/bash
# v2: predict on the ORIGINAL 9,382-pair test set (reconstructed under mapudungun_v2/data).
# Paper eval settings (predict_test.py): greedy, max_new_tokens=256, fp16, forced target tag.
#   DIRECTION=arn-es MODEL_PATH=... RUN_NAME=... sbatch --job-name=v2-pred-... configs/slurm/predict_test_v2.slurm
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs-1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/it238/projects/mapudungun/mapudungun-mt
TEST=/nobackup/archive/usr/it238/mapudungun_v2/data/blocks/cleaned/test/cleaned
OUT=/nobackup/archive/usr/it238/mapudungun_v2/predictions
TOK="${TOK:-standard}"
VC_TEST=/nobackup/archive/usr/it238/mapudungun_v2/data/blocks/morfessor_vc/test
if [ "$DIRECTION" = "arn-es" ]; then
    SRC=$TEST/src.txt; REF=$TEST/tgt.txt
    [ "$TOK" = "morfessor_vc" ] && SRC=$VC_TEST/src.txt   # segmented Mapudungun input
else
    SRC=$TEST/tgt.txt; REF=$TEST/src.txt                  # clean Mapudungun reference
fi
TOK_ARG=""; [ "$TOK" != "standard" ] && TOK_ARG="--tok-approach $TOK"   # es-arn VC: desegment output
.venv/bin/python -c "import transformers; assert transformers.__version__.startswith('4.'), transformers.__version__; print('transformers', transformers.__version__)"
.venv/bin/python scripts/eval/predict_test.py --direction "$DIRECTION" --model-path "$MODEL_PATH" \
    --run-name "$RUN_NAME" $TOK_ARG --src-file "$SRC" --ref-file "$REF" --out-dir "$OUT"
