# Bringing Mapudungun into the Modern MT Ecosystem

Isaac M. Thompson · Brandon M.A. Rogers · Eric K. Ringger

*Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)*

Code and model releases for bidirectional Mapudungun–Spanish (`arn↔es`) NLLB-200 fine-tuning under eight tokenization conditions.

Morfessor-VC begins with Morfessor segmentation and greedily removes boundaries when adjacent morphemes form an existing NLLB SentencePiece vocabulary item. This reduces added segmentation boundaries while retaining the base model tokenizer.

## Data

<!-- BEGIN GENERATED: data-summary -->
The corrected block-aligned dataset contains **55,452 training**, **1,581 development**, and **9,382 test** sentence pairs.
<!-- END GENERATED: data-summary -->

The experiments use the corpus introduced by Duan et al. (2020). The small tracked data sample is documented in [`data/README.md`](data/README.md) and retains its upstream noncommercial share alike license. The full corrected experimental splits are not committed to Git.

## Models

<!-- BEGIN GENERATED: release-status -->
The [Hugging Face collection](https://huggingface.co/collections/byumatrixlab/mapudungun-nllb-6a0b99ce4bbdd3b46531c6b4) currently contains **34 of 48** planned checkpoints. See [`results/release_status.json`](results/release_status.json) for the machine-checked list.
<!-- END GENERATED: release-status -->

Model repositories include condition metadata and any preprocessing assets required for inference. The model weights inherit the NLLB base model terms and the training data terms described on each model card.

## Corrected results

Mean sentence level chrF++ from [`results/paper_stats.json`](results/paper_stats.json):

<!-- BEGIN GENERATED: results -->
### arn→es

| Condition | 600M | 1.3B | 3.3B |
|---|---:|---:|---:|
| Standard BPE | **43.49** | **45.56** | 45.73 |
| Joint-5K BPE | 42.36 | 44.68 | 45.25 |
| Mono BPE | 41.72 | 43.99 | 44.85 |
| Optuna BPE | 42.41 | 44.67 | 45.21 |
| Morfessor | 43.09 | 45.40 | 45.51 |
| Morfessor-VC | 43.16 | 45.37 | **45.84** |
| Morfessor-BPE | 42.80 | 44.97 | 45.56 |
| UnigramLM | 42.18 | 44.64 | 45.07 |

### es→arn

| Condition | 600M | 1.3B | 3.3B |
|---|---:|---:|---:|
| Standard BPE | 39.87 | 41.94 | **42.89** |
| Joint-5K BPE | 38.17 | 41.06 | 42.30 |
| Mono BPE | 38.04 | 40.62 | 42.02 |
| Optuna BPE | 38.79 | 41.34 | 42.33 |
| Morfessor | **39.90** | 42.00 | 42.76 |
| Morfessor-VC | 39.89 | **42.04** | 42.77 |
| Morfessor-BPE | 38.74 | 41.25 | 42.40 |
| UnigramLM | 38.20 | 41.05 | 42.30 |
<!-- END GENERATED: results -->

The arn→es Standard BPE values use the corrected retrains on the recovered original data. The paper’s other conditions use the original published runs. See the v2 paper for the correction methodology and full BLEU, COMET, uncertainty, and human evaluation results.

## Setup

```bash
uv sync
source .venv/bin/activate
```

The training code requires Transformers 4.x because Transformers 5.x changes the NLLB tokenizer behavior used by these runs.

## Repository layout

```text
configs/              Slurm templates and cleaning configuration
data/                 Tracked sample and character inventories
poster/               AmericasNLP poster source and figures
release/              Canonical model release matrix
results/              Paper statistics and release audit JSON
scripts/analysis/     SSOT calculations, README rendering, and plots
scripts/data/         Corpus extraction and cleaning
scripts/eval/         Prediction and scoring
scripts/release/      Release audit, staging, and reconstruction tools
scripts/tokenization/ Tokenization conditions
scripts/training/     NLLB fine-tuning
```

## Reproduction

1. Prepare aligned `train`, `dev`, and `test` files under the layout accepted by `scripts/training/finetune_nllb.py`.
2. Apply the selected tokenizer in `scripts/tokenization/` to the Mapudungun side.
3. Fine-tune one size and direction with `scripts/training/finetune_nllb.py` or a template in `configs/slurm/`.
4. Generate predictions with `scripts/eval/predict_test.py` and compute paper statistics with the scripts listed in `scripts/analysis/paper_numbers.py`.
5. Regenerate public numeric claims with `python scripts/analysis/render_readme.py`.

Cluster paths in the Slurm files are examples and should be adapted to the local scheduler and storage layout.

## Licensing

Repository code is MIT licensed; see [`LICENSE`](LICENSE). Data under `data/cleaned_local/` is excluded from the MIT grant and is covered by [`data/LICENSE-DATA.md`](data/LICENSE-DATA.md). Model repositories state the applicable upstream model and data terms.

## Citation

```bibtex
@inproceedings{thompson2026mapudungun,
  title     = {Bringing {Mapudungun} into the Modern {MT} Ecosystem: Morphology-Aware Tokenization for {NLLB}-200 Fine-Tuning},
  author    = {Thompson, Isaac M. and Rogers, Brandon M.A. and Ringger, Eric K.},
  booktitle = {Proceedings of the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)},
  year      = {2026}
}
```

```bibtex
@inproceedings{duan-etal-2020-resource,
  title     = {A Resource for Computational Experiments on {Mapudungun}},
  author    = {Duan, Mingjun and Fasola, Carlos and Rallabandi, Sai Krishna and Vega, Rodolfo and Anastasopoulos, Antonios and Levin, Lori and Black, Alan W},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages     = {2872--2877},
  year      = {2020},
  url       = {https://aclanthology.org/2020.lrec-1.350/}
}
```
