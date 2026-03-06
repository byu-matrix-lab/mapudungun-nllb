#!/bin/bash
#SBATCH --job-name=morfessor-prep
#SBATCH --output=/home/it238/projects/mapudungun/mapudungun-mt/logs/morfessor-prep_%j.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=standby

cd /home/it238/projects/mapudungun/mapudungun-mt
uv run python scripts/tokenization/segment_morfessor.py \
    --data-root ~/nobackup/autodelete/mapudungun/data-processed \
    --model-out ~/nobackup/autodelete/mapudungun/morfessor_model.bin
