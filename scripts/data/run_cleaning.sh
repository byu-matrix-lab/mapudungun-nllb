#!/usr/bin/env bash
# run_cleaning.sh
#
# Run the BYU Matrix Lab data-cleaning-pipeline on both extraction approaches
# (lines and blocks), for all three splits (train / dev / test).
#
# Must be run from the project root:
#   bash scripts/data/run_cleaning.sh
#
# The pipeline is run from its own directory so it picks up config/arn-CL.yaml
# and config/es-ES.yaml from the lab installation.

set -euo pipefail

PIPELINE_DIR="/home/it238/groups/grp_mtlab/projects/data-cleaning/data-cleaning-pipeline"
DATA_DIR="/home/it238/nobackup/autodelete/mapudungun/data-processed"

for APPROACH in lines blocks; do
    for SPLIT in train dev test; do
        SRC="${DATA_DIR}/${APPROACH}/${SPLIT}.arn"
        TGT="${DATA_DIR}/${APPROACH}/${SPLIT}.es"
        OUT="${DATA_DIR}/${APPROACH}/cleaned"

        if [[ ! -f "$SRC" || ! -f "$TGT" ]]; then
            echo "Skipping ${APPROACH}/${SPLIT} — files not found"
            continue
        fi

        echo "=== Cleaning ${APPROACH}/${SPLIT} ==="
        cd "$PIPELINE_DIR"
        python3 pipeline.py \
            -t "${OUT}/${SPLIT}" \
            -srclang arn-CL \
            -tgtlang es-ES \
            -srcpath "$SRC" \
            -tgtpath "$TGT" \
            -d -v
        cd - > /dev/null
        echo
    done
done

echo "All done. Cleaned output in ${DATA_DIR}/{lines,blocks}/cleaned/"
