#!/bin/bash
# v2 option (c): Morfessor -> Morfessor-VC segmentation of the v2 data (regenerated train/dev,
# ORIGINAL test) with the paper's own scripts. CPU only.
#SBATCH --job-name=v2-segment
#SBATCH --partition=m9
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /home/it238/projects/mapudungun/mapudungun-mt
ROOT=/nobackup/archive/usr/it238/mapudungun_v2/data
.venv/bin/python -u scripts/tokenization/segment_morfessor.py --data-root $ROOT --approaches blocks \
    --model-out /nobackup/archive/usr/it238/mapudungun_v2/morfessor_model.bin
.venv/bin/python -u scripts/tokenization/segment_morfessor_vc.py --data-root $ROOT --approaches blocks
wc -l $ROOT/blocks/morfessor_vc/*/*.txt
sha256sum $ROOT/blocks/morfessor/*/*.txt $ROOT/blocks/morfessor_vc/*/*.txt >> $ROOT/SHA256SUMS
