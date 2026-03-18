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
- All completed successfully (~1:22 walltime for lines, ~2:00 for blocks)

**NLLB fine-tuning results (test set)**

| Job | Approach | Direction | chrF++ | BLEU |
|---|---|---|---|---|
| 10547333 | lines | arn→es | 44.39 | 18.10 |
| 10547334 | lines | es→arn | 42.40 | 16.37 |
| 10547335 | blocks | arn→es | 46.31 | 17.57 |
| 10547336 | blocks | es→arn | 44.18 | 16.05 |

Duan et al. (2020) reported 20.4 BLEU (arn→es) and 12.9 BLEU (es→arn) — different system, likely different BLEU tokenization.
- **es→arn**: we beat Duan et al. comfortably on both approaches
- **arn→es**: lines BLEU (18.10) is below their 20.4 — gap likely partly due to sacrebleu vs non-standard tokenization; chrF++ is more reliable anyway
- **Block-level outperforms line-level on arn→es chrF++** (46.31 vs 44.39) — full utterance context helps; reverses on es→arn (44.18 vs 42.40 but same direction)

**Next steps**: zero-shot NLLB-200 baseline (no fine-tuning), LLM baseline (Llama-3.1-8B few-shot), error analysis

---

## 2026-03-01 (continued)

**Zero-shot NLLB-200 results (test set, no fine-tuning)**

| Approach | Direction | chrF++ | BLEU |
|---|---|---|---|
| lines  | arn→es | 12.94 | 0.66 |
| lines  | es→arn | 6.48  | 0.18 |
| blocks | arn→es | 16.25 | 1.84 |
| blocks | es→arn | 10.96 | 0.98 |

Fine-tuning provides massive gains: +31 chrF++ (arn→es lines), +36 chrF++ (es→arn lines).
Makes sense — `arn_Latn` was not in NLLB's training data at all; the zero-shot model has no knowledge of Mapudungun.

**Llama-3.1-8B-Instruct 5-shot baseline**
- Model downloading in background (login node, nohup) — ~16GB, no quantization needed
- Script uses chat-template prompting + 5 dev examples; will submit 2 jobs when download done
- Jobs 10547432/33 (failed — 70B stub was not actually downloaded, only a cache entry existed)

**NLLB-1.3B fine-tuning queued**
- Jobs 10547536–39 (lines/blocks × arn→es/es→arn) — same hyperparams as 600M
- Output: `nobackup/mapudungun/models/nllb-1.3B-{approach}-{src}-{tgt}/`

**Zero-shot NLLB-1.3B queued**
- Jobs 10547541–44 (lines/blocks × arn→es/es→arn)

**Git**: fixed SLURM log naming (pass `--job-name` on CLI, `%x_%j` in output pattern);
parameterized MODEL env var in both finetune and zero_shot SLURM scripts.
`feature/data-pipeline` PR open on GitHub; `feature/nllb-finetuning` branch active.

**Llama-3.1-8B-Instruct jobs (lines) submitted**
- Jobs 10547548–49 (lines arn→es/es→arn) — running; output `results/llama/`
- Note: submitted before output-naming fix; filenames will be `llama_lines_{src}_{tgt}.json`

---

## 2026-03-01 (evening)

**NLLB result file naming fix**
- `zero_shot_nllb.py`: output stem now includes model tag (`zero_shot_{model_tag}_{approach}_{src}_{tgt}`)
- `llama_baseline.py`: output stem now model-tagged (`{model_tag}_{approach}_{src}_{tgt}`)
- `llama_baseline.slurm`: MODEL env var support added; output dir changed to `results/llm/`
- Zero-shot 600M results recovered from job logs and saved with correct `nllb-600M` naming;
  one file had been overwritten by 1.3B job (10547543 completed before rename)

**Zero-shot NLLB-1.3B results (partial — two jobs completed)**

| Approach | Direction | chrF++ | BLEU |
|---|---|---|---|
| blocks | arn→es | 17.22 | 1.77 |
| blocks | es→arn | 12.44 | 1.49 |

Remaining two (lines arn→es, lines es→arn) still running.

**NLLB-3.3B** — download complete (~6.6GB). Jobs queued:
- Fine-tuning: 10547568–71 (lines/blocks × arn→es/es→arn)
- Zero-shot:   10547572–75 (lines/blocks × arn→es/es→arn)

**Llama-3.1-8B-Instruct blocks jobs** — 10547576–77 (blocks arn→es/es→arn)

**Aya Expanse 8B** (`CohereForAI/aya-expanse-8b`) download started in background (~16GB)
- Specifically designed for multilingual/low-resource NLP — strong AmericasNLP framing comparison
- Will queue 4 jobs (lines+blocks × 2 directions) after download completes

**Related work decisions**
- Lira et al. (2025): 10K pairs, MarianMT, best 10.03 BLEU arn→es — cite, don't replicate (different splits)
- Peñas et al. (2023): 30K pairs + web data, reports 65.45 BLEU — likely tokenization inflation, not comparable

**Advisor's maxim**: "A good researcher has a clear hypothesis, clear experiments, and a deep supercomputer queue."

---

## 2026-03-02

**Aya Expanse 8B** (`CohereForAI/aya-expanse-8b`) — model was gated, accepted terms and redownloaded successfully. Jobs 10547580–83 submitted and completed.

**All overnight jobs completed** except NLLB-3.3B blocks fine-tune (OOM at batch=32).
- Fixed: `finetune_nllb.slurm` now supports `BATCH_SIZE`/`GRAD_ACCUM` env var overrides
- Resubmitted as 10551483–84 with `BATCH_SIZE=8 GRAD_ACCUM=16` (effective batch still 128) — running

**Complete results so far**

Fine-tuned NLLB (test set):

| Model | Approach | arn→es chrF++ | arn→es BLEU | es→arn chrF++ | es→arn BLEU |
|---|---|---|---|---|---|
| NLLB-600M | lines  | 44.39 | 18.10 | 42.40 | 16.37 |
| NLLB-600M | blocks | 46.31 | 17.57 | 44.18 | 16.05 |
| NLLB-1.3B | lines  | 45.42 | 18.83 | 43.90 | 17.68 |
| NLLB-1.3B | blocks | **48.59** | **19.52** | **47.32** | **18.38** |
| NLLB-3.3B | lines  | 44.88 | 17.87 | 43.68 | 17.43 |
| NLLB-3.3B | blocks | ⏳ pending | — | ⏳ pending | — |

Zero-shot NLLB (no fine-tuning):

| Model | Approach | arn→es chrF++ | es→arn chrF++ |
|---|---|---|---|
| NLLB-600M (distilled) | lines  | 12.94 | 6.48  |
| NLLB-600M (distilled) | blocks | 16.25 | 10.96 |
| NLLB-1.3B (distilled) | lines  | 13.62 | 8.12  |
| NLLB-1.3B (distilled) | blocks | 17.22 | 12.44 |
| NLLB-3.3B (non-distilled) | lines  | 2.96  | 1.99  |
| NLLB-3.3B (non-distilled) | blocks | 5.94  | 3.43  |

Note: 3.3B zero-shot is *worse* than distilled models — non-distilled model likely handles proxy
token initialization for `arn_Latn` differently. Worth discussing in paper.

LLM few-shot (5-shot):

| Model | Approach | arn→es chrF++ | es→arn chrF++ |
|---|---|---|---|
| Llama-3.1-8B-Instruct | lines  | 14.58 | 12.70 |
| Llama-3.1-8B-Instruct | blocks | 16.54 | 16.20 |
| Aya Expanse 8B        | lines  | 16.10 | 12.48 |
| Aya Expanse 8B        | blocks | **20.05** | **16.11** |

Key takeaways: fine-tuned NLLB-1.3B blocks is best at 48.59 chrF++ arn→es, beating all baselines
by ~28 chrF++ points. NLLB-1.3B > NLLB-600M > NLLB-3.3B after fine-tuning (larger model may need
lower LR / longer warmup). Aya Expanse outperforms Llama, consistent with multilingual focus.

**Script fixes**
- `zero_shot_nllb.py`: output stem now includes model tag (avoids overwriting across model sizes)
- `llama_baseline.py`: output stem is model-tagged
- `llama_baseline.slurm`: MODEL env var added; output dir → `results/llm/`
- 600M zero-shot files recovered from job logs after being overwritten by 1.3B run

---

## 2026-03-06

**NLLB-3.3B blocks fine-tune completed** (from previous session, jobs 10551483–84):
- arn→es: 49.25 chrF++ (dev), es→arn: 48.08 chrF++ (dev)
- These were **dev set** metrics — see "Test set evaluation" below for corrected numbers

**Results chart — `scripts/analysis/plot_results.py`**
- Created `notes/results_chrf.png`: dual-panel bar chart (arn→es left, es→arn right)
- Groups: Prior work (Duan 2020*, Lira 2025†), NLLB zero-shot (600M/1.3B/3.3B‡), 5-shot LLM (Llama 3.1 / Aya Exp.), Fine-tuned NLLB (600M/1.3B/3.3B)
- Footnotes: *Duan same conversations pre-cleaning; †Lira different 1,250-pair test set; ‡non-distilled model
- Chart will be updated once corrected test-set numbers are available for all model sizes

**Meeting with Dr. Brandon M. Rogers (BYU Spanish & Portuguese)**
- Showed demo translations — arn→es looked semantically solid; es→arn mostly Spanish output
- Potential roles: linguistic/error analysis, human evaluation (fluency + adequacy 1–5), grammar-augmented LLM prompting
- Demo translations script at `scripts/eval/demo_translations.py`

**Key discovery: `arn_Latn` absent from NLLB-200 tokenizer**
- Confirmed via `tokenizer.get_vocab()` inspection: `arn_Latn` resolves to UNK (id=3) in both 600M and 3.3B base models
- `finetune_nllb.py` already handles this correctly: adds `arn_Latn` as new token (id=256204), initializes from mean of `ayr_Latn`/`grn_Latn`/`quy_Latn`
- `demo_translations.py` was broken: it loaded the base tokenizer where `arn_Latn`=UNK, so `forced_bos_token_id`=3 during inference → model defaulted to Spanish
- **Fix**: load `NllbTokenizer` from base path + manually call `add_special_tokens({"additional_special_tokens": ["arn_Latn"]})` to reproduce training vocab (id=256204)

**Corpus code-switching analysis**
- Raw ELAN/CHAT files contain `<SPA>` and `<*SPA>` inline tags marking word-level code-switching (10,606 and 44,590 occurrences respectively across the corpus)
- Cleaning pipeline (`arn-CL.yaml`) strips all `<>` tags, so training data has no tag information
- fastText lid.176 word-level language ID on cleaned `src.txt` (Mapudungun side):
  - **~15% of words classified as Spanish** (consistent across blocks and lines)
  - **20–27% of lines have >20% Spanish words**
  - Remaining ~85% distributed across unrelated languages (Turkish, Italian, German, etc.) — fastText misclassification of native Mapudungun, which has no LID model
- Example high-CS lines: "si ese muy bueno ka fey.", "no si esa otra mujer ese pues de afuera" (entire discourse turns in Spanish)
- Conclusion: code-switching is authentic language use in these health consultation conversations, not noise

**Root cause of poor es→arn output**
- The model has correctly learned the code-switching distribution of the training corpus
- Many test references are themselves heavily Spanish-mixed → Spanish-heavy predictions can score well on code-switched references
- For "pure" Spanish inputs requiring "pure" Mapudungun output, the model fails (examples 1–4 in demo)
- Discussion with analysis: filtering is not clearly the right move; honest evaluation requires human annotation
- Decision: document the code-switching finding as a corpus analysis contribution, propose prefix-control and filtering as future work

**Morfessor tokenization study**
- `scripts/tokenization/segment_morfessor.py`: trains Morfessor 2.0 on 105,058 unique Mapudungun word types from training data
- Applies `@@` boundary markers: `ngütramkafiñ → ngütramka@@ fiñ`, `tukulpayayu → tukulpay@@ ayu`
- Short/unsegmented roots preserved: `inche`, `mapuche`, `kimün`, etc.
- Output ready at `data-processed/{blocks,lines}/morfessor/{train,dev,test}/src.txt`
- Two bugs fixed during prep: (a) `load_data()` expects `(count, word)` not `(word, count)`; (b) `get_segmentations()` returns generator, not len()-able
- Fine-tune jobs with `TOKENIZER_APPROACH=morfessor` submitted for 1.3B blocks:
  - 10672684: arn→es morfessor, 10672685: es→arn morfessor (PENDING)

**Test set evaluation — corrected numbers**
- Previous dev-set metrics for 3.3B blocks were misleading; ran `scripts/eval/predict_test.py` on full test set (9,382 pairs) with correct tokenizer

| Model | Direction | Test chrF++ | Dev chrF++ (prev reported) |
|---|---|---|---|
| NLLB-3.3B blocks | arn→es | **43.62** | 49.25 (dev) |
| NLLB-3.3B blocks | es→arn | **14.22** | 48.08 (dev, wrong tokenizer) |

- arn→es at 43.62 is still SOTA over Lira (30.30)
- es→arn at 14.22 is a major correction: 93% of sentences score <25 chrF++
  - Best predictions are all code-switched passthroughs (e.g. "Palo Trébol, palo Santo" → "Palo trebol, palo santo")
  - Score distribution: 44.4% at 0–10, 48.9% at 10–25
- 600M and 1.3B test-set predictions running (jobs 10672680–83); chart will be updated when done

**New evaluation infrastructure**
- `scripts/eval/predict_test.py`: runs any fine-tuned model on full test set, saves per-sentence JSON with chrF++ scores, reports distribution and best/worst examples
- `--model-path` and `--run-name` flags for running any model size
- Predictions saved to `nobackup/autodelete/mapudungun/predictions/`

**Jobs running / pending**
| Job | Task | Status |
|---|---|---|
| 10672680 | 600M arn→es predict | submitted |
| 10672681 | 600M es→arn predict | submitted |
| 10672682 | 1.3B arn→es predict | submitted |
| 10672683 | 1.3B es→arn predict | submitted |
| 10672684 | 1.3B morfessor arn→es fine-tune | PENDING |
| 10672685 | 1.3B morfessor es→arn fine-tune | PENDING |

**Next steps**
- Wait for predict jobs → update results chart with all test-set numbers
- Check morfessor fine-tune results when done
- Begin paper writing (~March 10 target)
- Set up Dr. Rogers human evaluation (50–100 sentence fluency+adequacy 1–5 for arn→es)
- Move best model checkpoints out of autodelete

---

## 2026-03-09

**Critical bug found and fixed: all es→arn models were trained on the wrong direction**

Root cause: `load_split()` in `finetune_nllb.py` always loaded `src.txt` (Mapudungun) as
source and `tgt.txt` (Spanish) as target, regardless of `--src`/`--tgt` arguments. For
es→arn runs, the model saw Mapudungun input (labeled spa_Latn) → Spanish output (labeled
arn_Latn). The trainer reported dev chrF++ ≈ 47 because it evaluated the model on the
same wrong task (arn→es). When `predict_test.py` correctly tested Spanish→Mapudungun, it
got 14.22 — not from corpus code-switching, but because the model never learned that task.

Fix: `load_split()` now takes a `src` parameter and swaps `src.txt`/`tgt.txt` when
`src == "es"`. All es→arn models must be retrained.

**Retraining jobs submitted (corrected es→arn)**

| Job | Model | Direction | Notes |
|---|---|---|---|
| 10723296 | NLLB-600M | blocks es→arn | corrected |
| 10723297 | NLLB-1.3B | blocks es→arn | corrected |
| 10723298 | NLLB-3.3B | blocks es→arn | corrected, BATCH_SIZE=8 |
| 10723299 | NLLB-1.3B | blocks es→arn morfessor | corrected |

Output dirs use same naming convention as before (outputs will overwrite old buggy models
in `-v2`-style naming since the output dir is derived from model+approach+src+tgt).

**Morfessor arn→es evaluation**

- `predict_test.py` updated with `--morfessor` flag: loads pre-segmented test src from
  `blocks/morfessor/test/src.txt`, scores against clean Spanish refs
- For es→arn morfessor: desegments model output before scoring against Mapudungun refs
- `predict_test.slurm` updated with `MORFESSOR` env var
- Morfessor arn→es predict job submitted: 10723302

**comet_score.py overhauled**

- Now reads `*_preds.txt` / `*_results.json` (predict_test.py format)
- Reads src/ref/pred directly from per-sentence JSON (no --data-dir needed)
- Writes per-run `*_comet.json` summaries alongside predictions
- `comet_score.slurm` updated: RESULTS_DIR → predictions/; dropped --data-dir
- COMET job submitted: 10723307

**Results chart updated**

- `plot_results.py`: fine-tuned numbers updated to corrected test-set values
- es→arn column shows 14.40/14.25/14.22 (still the wrong-direction results, pending retrain)
- Will be updated again once corrected retrain jobs complete

**Current state of results (arn→es test set, blocks)**

| Model | arn→es chrF++ | es→arn chrF++ |
|---|---|---|
| NLLB-600M standard | 35.71 | pending retrain |
| NLLB-1.3B standard | 42.24 | pending retrain |
| NLLB-3.3B standard | 43.62 | pending retrain |
| NLLB-1.3B morfessor | 47.31 (trainer) | pending retrain |

Note: the previously reported "es→arn morfessor 47.36" result was the same bug — the
model was trained arn→es with wrong language labels, not es→arn. Discarded.

**Git**: commit `ff7bff3` on `feature/nllb-finetuning`

---

## 2026-03-16

**Root cause found for 1.3B and 3.3B es→arn bad predictions**

The corrected es→arn models (1.3B, 3.3B) saved their weights in sharded format
(`model-00001-of-N.safetensors` + `model.safetensors.index.json`) because they
exceeded the single-file threshold. However, the original buggy models from March 1/2
had written a single-file `model.safetensors` into the same `best/` directory and were
never deleted when the corrected job overwrote all other files.

`transformers.from_pretrained` checks for `model.safetensors` (single file) *before*
`model.safetensors.index.json` (sharded) in its elif chain. When both are present, it
loads the old single-file weights and silently ignores the shards. The morfessor 1.3B
model was unaffected because it had no prior run in that directory (no stale single file).

**Fix**: renamed `model.safetensors` → `model.safetensors.OLD_BUGGY_ARN2ES` in both
`nllb-1.3B-blocks-es-arn/best/` and `nllb-200-3.3B-blocks-es-arn/best/`.

**Corrected predict-test results**

| Model | arn→es chrF++ | es→arn chrF++ |
|---|---|---|
| NLLB-600M standard | 34.90 | 39.87 |
| NLLB-1.3B standard | 40.64 | pending (job 10912642) |
| NLLB-3.3B standard | 42.86 | 42.89 |
| NLLB-1.3B morfessor | 45.40 | 42.00 |

Note: 600M es→arn 39.87 was already correct before today — its `best/` dir had no
stale single file. The March 9 comet.json (chrf=12.99) was stale from the pre-retrain run.

**Next steps**: wait for 1.3B es→arn (job 10912642), run COMET on all updated
predictions, update `plot_results.py`, begin paper draft.

---

## 2026-03-16 (evening)

**Full experiment matrix launched — all conditions queued**

Closed remaining prediction gaps and launched all new tokenization conditions.

**New predict-test results (from jobs 10923065–67):**

| Model | arn→es chrF++ | es→arn chrF++ |
|---|---|---|
| NLLB-600M standard | 34.90 | 39.87 |
| NLLB-1.3B standard | 40.64 | 41.94 |
| NLLB-3.3B standard | **42.85** | 42.89 |
| NLLB-600M morfessor | **43.09** | 39.90 |
| NLLB-1.3B morfessor | 45.40 | 42.00 |
| NLLB-3.3B morfessor | pending (jobs 10925593/94) | pending |

Notable: 600M+Morfessor (43.09) beats 3.3B standard (42.85). Smaller model with better tokenization outperforms a model 5× larger.

3.3B morfessor trainer-reported dev scores (from jobs 10912957/58):
- arn→es: 48.83 chrF (best epoch 8) — training done, predict pending
- es→arn: 46.13 chrF (best epoch 9) — training still running (job 10912958)

**Three new tokenization scripts written:**
- `scripts/tokenization/segment_bpe.py` — Duan 5K joint BPE + large arn-only BPE (SentencePiece backend, @@ output convention)
- `scripts/tokenization/segment_cascade.py` — NLLB-constrained Morfessor+BPE cascade: merges adjacent morfessor morphemes where the concatenation is a single NLLB SentencePiece token. Train-set: 87,382 → 85,167 @@ markers (2.5% reduction).
- Data generated for blocks: `duan_bpe/`, `large_bpe/`, `cascade/` in data-processed/blocks/

**Training + inference pipeline updated:**
- `finetune_nllb.py`: extended `--tokenizer-approach` choices to include `duan_bpe`, `large_bpe`, `cascade`; desegmentation condition generalized to all non-standard approaches
- `predict_test.py`: added TEST_DATA entries for all 3 new conditions (both directions); added `--tok-approach` flag (backwards-compatible with `--morfessor`)

**All remaining training jobs submitted with full dep chains:**

| Jobs | Condition | Status |
|---|---|---|
| 10925585/86 | cascade 1.3B arn↔es | running |
| 10925587–92 | duan_bpe 600M/1.3B/3.3B arn↔es | running |
| 10925599/600 | cascade 600M arn↔es | running |
| 10925601/602 | cascade 3.3B arn↔es | running |
| 10925603–608 | large_bpe 600M/1.3B/3.3B arn↔es | running |
| 10925609–626 | predict_test for all above (dep on training) | pending |
| 10925627 | final COMET sweep (dep on all predicts) | pending |
| 10925629 | COMET on existing predictions (no dep) | running |

**Decision: blocks approach only.** Lines approach dropped — blocks outperforms lines and is the focus for the paper.

**Next session priorities:**
1. Check 3.3B morfessor predict results when jobs 10925593/94 complete
2. Code-switching quantification script (`scripts/analysis/codeswitching_stats.py`)
3. Fertility analysis across all 5 tokenization conditions (once cascade/BPE data exists)
4. Email Dr. Rogers about human evaluation schedule
5. Begin paper draft (intro + related work can be written now)

---

## 2026-03-17

**All jobs completed overnight — full results table finalized**

Every training + predict + COMET job from the 2026-03-16 launch finished by morning.

**Complete results (mean per-sentence chrF++ on test set, n=9,382):**

| Condition | 600M arn→es | 600M es→arn | 1.3B arn→es | 1.3B es→arn | 3.3B arn→es | 3.3B es→arn |
|---|---|---|---|---|---|---|
| Standard BPE | 34.90 | 39.87 | 40.64 | 41.94 | 42.85 | 42.89 |
| Morfessor | 43.09 | 39.90 | 45.40 | 42.00 | 45.51 | 42.76 |
| Cascade (NLLB-constrained) | **43.16** | 39.89 | 45.37 | **42.04** | **45.84** | 42.77 |
| Duan 5K BPE | 42.36 | 38.17 | 44.68 | 41.06 | 45.25 | 42.30 |
| Large language-specific BPE | 41.72 | 38.04 | 43.99 | 40.62 | 44.85 | 42.02 |

**Key findings:**

1. **Cascade is best for arn→es at all model sizes.** 3.3B cascade (45.84) is the overall best, beating 3.3B standard by +3.0 points. The NLLB-constrained Morfessor+BPE cascade outperforms plain Morfessor at 3.3B and 600M.

2. **600M cascade (43.16) beats 3.3B standard (42.85).** Tokenization improvement bridges a 5× model size gap. Strong efficiency argument for the paper.

3. **es→arn is nearly flat across all tokenization conditions.** Standard BPE is essentially as good as any morphological pre-segmentation method for es→arn. Best improvement is only +0.15 (1.3B cascade). This asymmetry makes linguistic sense: segmenting the Mapudungun *source* helps the encoder handle polysynthetic morphology; the *target-side* generation of Mapudungun is controlled by the decoder regardless of tokenization strategy.

4. **Duan BPE and Large BPE hurt es→arn.** All BPE-only variants are worse than standard for es→arn generation. Suggests that further fragmentation of the target-side Mapudungun tokens is harmful for decoding.

5. **Morfessor provides most of the gain; cascade provides a small additional improvement for arn→es.** At 1.3B, Morfessor (45.40) slightly outperforms cascade (45.37) — within noise. The cascade's advantage is clearest at 3.3B (+0.33 over Morfessor).

**Main narrative:** Morphological pre-segmentation of the Mapudungun source is the key lever for polysynthetic MT. Cascade (Morfessor → NLLB-constrained BPE) is the best instantiation of this idea, and its efficiency gain (600M cascade > 3.3B standard) is the headline practical result.

**Zero-shot NLLB baselines and LLM baselines:**

| System | arn→es | es→arn |
|---|---|---|
| Zero-shot NLLB-600M | 12.86 | 7.37 |
| Zero-shot NLLB-1.3B | 13.43 | 9.91 |
| Zero-shot NLLB-3.3B | 13.19 | 9.42 |
| Llama 3.1 8B (prompted) | 13.96 | 11.32 |
| Aya Expanse 8B (prompted) | **15.92** | **12.36** |

Fine-tuning provides ~3× absolute gain over all baselines. 600M cascade (43.16) is nearly 3× Aya's best (15.92).

**Fertility analysis across all 5 tokenization conditions** (on test set, Mapudungun source):

| Condition | Pre-seg tok/word | NLLB tok/orig word |
|---|---|---|
| Standard BPE | 1.00 | 2.30 |
| Morfessor | 1.27 | 2.87 |
| Cascade (NLLB-constrained) | 1.25 | 2.82 |
| Duan 5K BPE | 1.71 | 3.83 |
| Large BPE | 2.08 | 4.77 |

Key insight: Morfessor and cascade increase NLLB token count per word (2.82–2.87 vs 2.30 for standard) yet *improve* performance — the morpheme boundaries give the encoder better representations despite more tokens. Duan and Large BPE push beyond a threshold (3.83, 4.77) where over-segmentation degrades performance. The cascade's NLLB-constrained merges produce slightly fewer NLLB tokens than plain Morfessor (2.82 vs 2.87), confirming the design reduces double-tokenization.

**Code-switching analysis** — `scripts/analysis/codeswitching_stats.py`
Full results in `notes/codeswitching_analysis.txt`.

- **54.7%** of test sentences contain ≥1 Spanish word (fastText word-level LID)
- **26.8%** heavily code-switched (>20% Spanish words)
- Correlation with chrF++ is *positive*: Spearman r = +0.051 — more Spanish → slightly better performance
- Mean chrF++ by CS level: No CS = 45.75, Light CS = 45.11, Heavy CS = 46.77
- Interpretation: Spanish borrowed words anchor the encoder; pure Mapudungun sentences require full morphological handling with no crutch
- Failure modes: hallucination loops on rare morphemes, proper nouns, short utterances
- Caveat: fastText mis-tags some Mapudungun discourse markers (Fey, eymi) as Spanish — true CS rate somewhat below 54.7%

**COMET**: 42/42 results files have corresponding comet.json files — full coverage.

**Emailed Dr. Rogers** requesting ~50-sentence arn→es human evaluation (fluency + adequacy, 1–5 scale) by April 7.

**All experiments complete. Remaining work is the paper.**

Paper sections that can be written now without waiting for anything:
- Introduction
- Related work
- Data section
- Methods section
- Results (all numbers in hand)
- Analysis (code-switching, fertility, failure modes)

Human evaluation (Dr. Rogers) will populate a small table in the analysis section when it arrives.

---

## 2026-03-17

**All experiments complete. Full results compiled.**

Zero-shot NLLB and LLM baselines finished:

| System | arn→es | es→arn |
|---|---|---|
| Zero-shot NLLB-600M | 12.86 | 7.37 |
| Zero-shot NLLB-1.3B | 13.43 | 9.91 |
| Zero-shot NLLB-3.3B | 13.19 | 9.42 |
| Llama 3.1 8B (prompted) | 13.96 | 11.32 |
| Aya Expanse 8B (prompted) | 15.92 | 12.36 |

Fine-tuning provides ~3× absolute gain over all baselines. 600M cascade (43.16 chrF++) is nearly 3× Aya's best (15.92).

**Corpus-level metrics compiled** (`results/metrics_table.tsv`) — chrF++, BLEU, COMET for all 42 conditions. Best result: 3.3B cascade arn→es at **46.08 chrF++ / 18.94 BLEU / 0.6653 COMET**.

**Bootstrap significance testing** (`scripts/eval/significance_test.py`, n=10,000 resamples):

Key results (arn→es):
- Cascade > standard: Δ=+2.99, p<0.0001 ✓
- Cascade > morfessor: Δ=+0.34, p=0.0005 ✓ (only at 3.3B; 1.3B is a tie: Δ=-0.03, p=0.70)
- Morfessor > standard: Δ=+2.66, p<0.0001 ✓
- 600M cascade > 3.3B standard: Δ=+0.31, p=0.037 ✓ — tokenization > scale, statistically confirmed
- Cascade > Duan BPE: Δ=+0.59, p<0.0001 ✓
- Cascade > Large BPE: Δ=+1.00, p<0.0001 ✓

es→arn: no tokenization comparison is significant — standard BPE is as good as any alternative.

**Figures generated:**
- `notes/results_chrf.png` — overview bar chart (prior work / zero-shot / LLM / fine-tuned)
- `notes/results_tokenization.png` — tokenization ablation grouped bar chart (5 conditions × 3 sizes)
- `notes/results_table.png` — compact lab-meeting table

**Readable output files for best model:**
- `notes/3.3B-cascade-arn-es_readable.txt`
- `notes/3.3B-cascade-es-arn_readable.txt`

**Remaining work: paper writing + human evaluation (awaiting Dr. Rogers reply).**

<!-- Add new entries below as work progresses -->
