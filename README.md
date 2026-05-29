# Bringing Mapudungun into the Modern MT Ecosystem

**Isaac M. Thompson¹, Brandon M.A. Rogers², Eric K. Ringger¹**
¹Department of Computer Science, Brigham Young University · ²Department of Spanish & Portuguese, Brigham Young University
Published at the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)

---

## Overview

Mapudungun (ISO 639-3: `arn`) is a polysynthetic language spoken by ~200,000 Mapuche people in Chile and Argentina. This project fine-tunes NLLB-200 (600M, 1.3B, and 3.3B) bidirectionally (arn↔es) on the Duan et al. (2020) 266K-pair corpus and conducts a systematic study of eight tokenization strategies suited to polysynthetic morphology.

The key contribution is **Morfessor-VC**: Morfessor segmentation constrained to NLLB's pretrained SentencePiece vocabulary, eliminating double-tokenization artifacts. For arn→es, morphology-aware tokenization with the 600M model matches Standard BPE with a model 5× larger.

---

## Models

All 48 fine-tuned models (8 tokenization conditions × 3 sizes × 2 directions) are available on HuggingFace:
[byumatrixlab/mapudungun-nllb](https://huggingface.co/collections/byumatrixlab/mapudungun-nllb-6a0b99ce4bbdd3b46531c6b4)

---

## Results (chrF++)

| Condition | 600M arn→es | 1.3B arn→es | 3.3B arn→es |
|-----------|------------|------------|------------|
| Standard BPE | 34.9 | 40.6 | 42.9 |
| Joint-5K BPE | 42.4 | 44.7 | 45.3 |
| Mono BPE | 41.7 | 44.0 | 44.9 |
| Optuna BPE | 42.4 | 44.7 | 45.2 |
| Morfessor | 43.1 | 45.4 | 45.5 |
| **Morfessor-VC** | **43.2** | **45.4** | **45.8** |
| Morfessor-BPE | 42.8 | 45.0 | 45.6 |
| UnigramLM | 42.2 | 44.6 | 45.1 |

---

## Repository Structure

```
mapudungun-mt/
├── data/
│   ├── cleaned_local/       # Local Mapuche transcript data (~5.9K pairs)
│   └── char_sets/           # Character frequency analysis for arn
├── scripts/
│   ├── data/                # Corpus extraction and cleaning
│   ├── tokenization/        # Tokenizer training (BPE, Morfessor, Morfessor-VC, UnigramLM)
│   ├── training/            # NLLB fine-tuning
│   ├── eval/                # Prediction, chrF++, COMET, significance testing
│   ├── analysis/            # Fertility, code-switching, result plots
│   └── upload_to_hf.py      # Upload models to HuggingFace
├── configs/
│   ├── cleaning/            # arn-CL.yaml — Mapudungun-specific cleaning rules
│   └── slurm/               # SLURM job templates (BYU supercomputer)
└── results/
    └── metrics_table.tsv    # Full chrF++ results for all conditions
```

The raw 266K corpus (Duan et al. 2020) and model checkpoints are not included; see the HuggingFace collection for models.

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

On compute nodes with `HF_HUB_OFFLINE=1`: cache models in `$HF_HOME` before submitting SLURM jobs.

---

## Reproducing Results

**1. Segment training data** (example: Morfessor-VC)
```bash
python scripts/tokenization/segment_morfessor_vc.py \
  --data data/cleaned_local/cleaned/ --output data-processed/blocks/morfessor_vc/
```

**2. Fine-tune**
```bash
sbatch configs/slurm/finetune_nllb.slurm  # set APPROACH, MODEL_TAG, SRC, TGT
```

**3. Predict and score**
```bash
sbatch configs/slurm/predict_test.slurm
python scripts/eval/significance_test.py
```

---

## Citation

```bibtex
@inproceedings{thompson2026mapudungun,
  title     = {Bringing {Mapudungun} into the Modern {MT} Ecosystem: Morphology-Aware Tokenization for {NLLB}-200 Fine-Tuning},
  author    = {Thompson, Isaac M. and Rogers, Brandon M.A. and Ringger, Eric K.},
  booktitle = {Proceedings of the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)},
  year      = {2026},
}
```

Data citation:
```bibtex
@inproceedings{duan2020mapudungun,
  title     = {A Resource for Computational Experiments on {Mapudungun}},
  author    = {Duan, Lindia and others},
  booktitle = {Proceedings of LREC 2020},
  pages     = {2872--2877},
  year      = {2020},
}
```
