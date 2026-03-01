# Research Log — Mapudungun MT (AmericasNLP 2026)

Paper: "Bringing Mapudungun into the Modern MT Ecosystem"
Venue: AmericasNLP 2026 @ ACL — submission deadline **April 15, 2026**
Authors: Isaac Thompson, Brandon M. Rogers

---

## 2026-02-27

**Repository setup**
- Created `mapudungun-mt` repo; established branching strategy: `main` (stable) / `develop` (integration) / `feature/*`
- Folder structure: `scripts/data/`, `scripts/training/`, `configs/cleaning/`, `configs/slurm/`, `notes/`, `results/`

**Data extraction — `scripts/data/extract_parallel_duan.py`**
- Corpus: Duan et al. (2020) AVENUE corpus, stored at `/home/it238/nobackup/autodelete/mapudungun/mapudungun-corpus/`
  - 343 `translation-clean/*.txt` files; 10 files present in official split lists but absent from the GitHub release (all `nmlch-*` prefixed — ~96% coverage, explainable in paper)
  - Official train/dev/test split lists from `dataset_splits/mt/`: train=285 files, dev=12, test=46
- Two extraction approaches implemented:
  - `--approach lines`: positional M:/C: line pairing (~256K pairs)
  - `--approach blocks`: concatenate all M: lines per utterance block, likewise for C: (~88K pairs)
- No cleaning in extraction; all normalization deferred to the lab pipeline
- Output pair counts after extraction:

| Approach | Train | Dev | Test | Total |
|---|---|---|---|---|
| lines | 213,880 | 7,263 | 34,923 | 256,066 |
| blocks | 73,020 | 2,138 | 12,430 | 87,588 |

---

## 2026-02-28

**Corpus analysis — `scripts/data/analyze_corpus.py`**
- Ran character frequency and noise analysis on extracted line-level pairs
- Key findings that informed cleaning config design:
  - ü: 121K occurrences; ñ: 59K — both valid Mapudungun chars, must be preserved
  - 7.7% duplicate pairs; 4.5% source == target (both always-on in lab pipeline)
  - 108 incomplete ELAN/CHAT tags (`SPA>`, `Noise>`, etc.) — need regex to strip
  - 2,562 overlap markers (`+/.../+`) — strip marker chars, keep enclosed text
  - 602 `%`-code lines (CHAT transcription tier codes) — strip
  - Anomalous non-ASCII chars confirmed absent from valid Mapudungun: ç, ö, ¥, grave accents

**Cleaning config — `configs/cleaning/arn-CL.yaml`**
- Deployed to lab pipeline at `/home/it238/groups/grp_mtlab/projects/data-cleaning/data-cleaning-pipeline/config/arn-CL.yaml`
- `normalize_regex_patterns`:
  - Strip full ELAN/CHAT inline tags (`<*SPA>`, `<Noise>`, etc.)
  - Strip incomplete tags missing opening `<` (`SPA>`, `Noise>`)
  - Strip `+/` and `/+` overlap markers; strip `-/` and `/-` uncertainty markers (keep enclosed text)
  - Strip `%\S*` CHAT tier codes
  - Strip backslashes, null bytes; normalize tabs and multiple spaces
- `allowed_characters`: full printable ASCII (32–126) + ¡¿ÁÉÍÑÓÚÜáéíñóúü
- **Critical overrides** of `default.yaml`:
  - `min_num_words: 1` (default is 3 — would wrongly discard single-word Mapudungun clauses, polysynthetic language)
  - `remove_long_words: False` (default removes words >30 chars — Mapudungun compounds regularly exceed this)
- First cleaning run used default `min_num_words: 3` before override was discovered — caused 34.84% deletion on lines/train (47K pairs lost). Fixed and rerun.

**Cleaning pipeline run — `scripts/data/run_cleaning.sh`**
- Runs `pipeline.py` from the lab pipeline directory (so `config/` is found correctly)
- Iterates all approaches × splits: lines/{train,dev,test}, blocks/{train,dev,test}
- Output at `/home/it238/nobackup/autodelete/mapudungun/data-processed/{approach}/cleaned/{split}/cleaned/src.txt|tgt.txt`
- Final retention after fix:

| Approach | Train | Dev | Test | Total | Retention |
|---|---|---|---|---|---|
| lines | 170,167 | 5,873 | 27,950 | 204,000 | ~80% |
| blocks | 55,452 | 1,581 | 9,382 | 66,415 | ~76% |

---

## 2026-03-01

**Key discovery: `arn_Latn` absent from NLLB-200**
- Confirmed by inspecting `tokenizer.json` in the cached `facebook/nllb-200-distilled-600M` model
- NLLB-200 has 202 language tokens; Mapudungun is not among them
- **Decision**: add `arn_Latn` as a new special token; initialize its embedding from the mean of the three closest indigenous Latin-script languages present in NLLB:
  - `ayr_Latn` (Aymara, id=256018)
  - `grn_Latn` (Guaraní, id=256063)
  - `quy_Latn` (Quechua, id=256144)
- `spa_Latn` (id=256161) used as `forced_bos_token_id` for arn→es; `arn_Latn` (id=256204 after resize) for es→arn
- This is a meaningful paper contribution — explicit justification for proxy language choice needed in methods

**Training script — `scripts/training/finetune_nllb.py`**
- HuggingFace `Seq2SeqTrainer` wrapper; handles both directions via `--src`/`--tgt`
- Adds `arn_Latn` token and initializes embedding if absent
- Evaluates chrF++ (primary) and BLEU (secondary) via sacrebleu after each epoch
- Early stopping with patience=3 on dev chrF++
- Updated for transformers 4.46+ API:
  - `evaluation_strategy` → `eval_strategy`
  - `tokenizer=` → `processing_class=` in `Seq2SeqTrainer`
  - `model.config.forced_bos_token_id` → `model.generation_config.forced_bos_token_id`

**SLURM script — `configs/slurm/finetune_nllb.slurm`**
- `--qos=matrix` (A100, dwmatrix partition), `--gpus=1`, 64G RAM, 8 CPUs, 24h walltime
- Parameterized via `APPROACH`, `SRC`, `TGT` env vars set before `sbatch`
- Hyperparameters: batch=32, grad_accum=4 (effective 128), lr=3e-5, warmup=1000, max_length=128, epochs=10, num_beams=4

**Jobs submitted (first attempt — failed)**
- 10547306 `lines arn→es`, 10547311 `lines es→arn`, 10547312 `blocks arn→es`, 10547313 `blocks es→arn`
- **Failure**: crashed at epoch 1 eval — `model.config.forced_bos_token_id` deprecated in newer transformers; must use `model.generation_config.forced_bos_token_id`
- Training itself was working (loss on 10547306: 15.44 → 13.16 → 11.66 → 10.52 across first 10% of training)

**Jobs resubmitted after fix**
- 10547333 `lines arn→es`, 10547334 `lines es→arn`, 10547335 `blocks arn→es`, 10547336 `blocks es→arn`
- All currently running/queued on A100 (dwmatrix)

---

<!-- Add new entries below as work progresses -->
