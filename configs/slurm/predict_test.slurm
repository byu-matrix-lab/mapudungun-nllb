#!/bin/bash
#SBATCH --job-name=predict-test
#SBATCH --output=logs/predict-test_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --qos=standby

DIRECTION="${DIRECTION:-es-arn}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_NAME="${RUN_NAME:-}"
MORFESSOR="${MORFESSOR:-}"       # legacy: set to any non-empty value for morfessor
TOK_APPROACH="${TOK_APPROACH:-}" # preferred: morfessor | duan_bpe | large_bpe | cascade

cd /home/it238/projects/mapudungun/mapudungun-mt
export HF_HUB_OFFLINE=1

EXTRA_ARGS=""
[ -n "$MODEL_PATH"   ] && EXTRA_ARGS="$EXTRA_ARGS --model-path $MODEL_PATH"
[ -n "$RUN_NAME"     ] && EXTRA_ARGS="$EXTRA_ARGS --run-name $RUN_NAME"
[ -n "$TOK_APPROACH" ] && EXTRA_ARGS="$EXTRA_ARGS --tok-approach $TOK_APPROACH"
[ -n "$MORFESSOR"    ] && EXTRA_ARGS="$EXTRA_ARGS --morfessor"

echo "Running prediction: direction=$DIRECTION model=$MODEL_PATH run=$RUN_NAME morfessor=$MORFESSOR"
uv run python scripts/eval/predict_test.py --direction "$DIRECTION" $EXTRA_ARGS
