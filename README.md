# Bringing Mapudungun into the Modern MT Ecosystem

**Isaac Thompson & Dr. Brandon M. Rogers** (BYU Spanish & Portuguese)
Submitted to AmericasNLP 2026 @ ACL — deadline April 15, 2026

---

## Overview

Mapudungun (ISO 639-3: `arn`) is a polysynthetic language spoken by 200,000+ Mapuche people in Chile and Argentina. Despite a 266,300-pair parallel corpus (Duan et al., LREC 2020), it remains absent from all major MT systems and no modern multilingual model has been fine-tuned on it.

This project:
1. Fine-tunes `facebook/nllb-200-distilled-600M` bidirectionally (arn↔es) on the full 266K corpus
2. Conducts a systematic **tokenization study** comparing four strategies suited to polysynthetic morphology
3. Establishes **LLM prompting baselines** (open-source, on-cluster) for comparison

---

## Repository Structure

```
mapudungun-mt/
├── proposal.md              # Full research proposal
├── requirements.txt         # Python dependencies
├── data/
│   ├── splits/              # Train/dev/test index files (240K/13K/13K)
│   ├── cleaned_local/       # Cleaned local Mapuche transcript data (~5.9K pairs)
│   └── char_sets/           # Character frequency analysis for arn-CL
├── scripts/
│   ├── data/                # Extraction and data preparation scripts
│   ├── tokenization/        # Tokenizer training and fertility comparison
│   ├── training/            # NLLB fine-tuning and LLM baseline scripts
│   └── evaluation/          # chrF++ and BLEU scoring
├── configs/
│   ├── cleaning/            # arn-CL.yaml — Mapudungun-specific cleaning rules
│   └── slurm/               # SLURM job templates
├── results/                 # Experiment outputs (metrics, logs)
└── notebooks/               # Analysis, figures, error analysis
```

**Large files not in this repo:**
- Raw corpus (`Mapudungun Data/`) — stored in `/nobackup/autodelete/Mapudungun Project/`
- Model checkpoints — stored in `/nobackup/` or `results/` (symlinked)
- Cleaned 266K data — stored in `/nobackup/autodelete/Mapudungun Project/`

---

## Data Cleaning Pipeline

Uses the lab's canonical pipeline at `/home/it238/groups/grp_mtlab/projects/data-cleaning/data-cleaning-pipeline/` with the Mapudungun-specific config at `configs/cleaning/arn-CL.yaml`.

```bash
python3 /home/it238/groups/grp_mtlab/projects/data-cleaning/data-cleaning-pipeline/pipeline.py \
  -t data/cleaned_duan/ \
  -srclang arn-CL -tgtlang es-ES \
  -srcpath [PATH_TO_ARN] -tgtpath [PATH_TO_ES] \
  -d -v
```

---

## Tokenization Study

Four strategies compared by **fertility** (tokens/word, lower is better for polysynthetic languages):

| ID | Strategy | Tool |
|----|----------|------|
| (a) | Duan et al. 5K joint BPE | `sentencepiece` |
| (b) | Larger language-specific BPE (16K, 32K) | `sentencepiece` |
| (c) | Morfessor segmentation | `morfessor` |
| (d) | Linguistically-informed segmentation | Collaboration with Dr. Rogers |

```bash
python scripts/tokenization/compare_tokenizers.py --data data/splits/ --output results/tokenization/
```

---

## Training

Fine-tunes NLLB-200-distilled-600M using HuggingFace `Seq2SeqTrainer`. Submit via SLURM:

```bash
sbatch configs/slurm/finetune.sh --tokenizer bpe5k --direction arn-es
```

Results are saved to `results/[date]_[tokenizer]_[direction]/`.

---

## Evaluation

Primary metric: **chrF++** (via `sacrebleu`). Secondary: BLEU, tokenizer fertility.

```bash
python scripts/evaluation/score.py --hyp results/.../output.txt --ref data/splits/test.es
```

---

## Branching Strategy

- `main` — stable, tagged releases only
- `develop` — integration branch; all PRs merge here first
- `feature/data-pipeline` — data cleaning and preparation
- `feature/tokenization` — tokenization comparison
- `feature/nllb-finetuning` — model fine-tuning
- `feature/llm-baseline` — LLM prompting experiments
- `feature/evaluation` — metrics and analysis

All feature branches require a pull request and Isaac's review before merging to `develop`.

---

## Environment Setup

```bash
uv sync
source .venv/bin/activate
```

On compute nodes (`HF_HUB_OFFLINE=1`): ensure models are cached in `$HF_HOME` before submitting jobs.

---

## Citation

If using the Duan et al. corpus:
> Duan et al. (2020). A parallel corpus for low-resource Mapudungun. LREC 2020.
