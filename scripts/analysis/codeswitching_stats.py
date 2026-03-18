#!/usr/bin/env python3
"""
Code-switching analysis on the test set.

For each test sentence (Mapudungun source), uses fastText word-level LID to
compute the fraction of Spanish tokens. Correlates this with per-sentence
chrF++ from the best model (3.3B cascade, arn→es). Reports:
  - % sentences with ≥1 Spanish word
  - % sentences with >20% Spanish words (heavy CS)
  - Pearson/Spearman correlation between Spanish word% and chrF++
  - Best/worst examples split by CS level
  - Bucket analysis: low / medium / high CS → mean chrF++
"""

import json
import re
from pathlib import Path

import fasttext
import numpy as np
from scipy import stats

LID_MODEL   = "/home/it238/nobackup/autodelete/mapudungun/lid.176.bin"
TEST_SRC    = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/src.txt")
TEST_TGT    = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/tgt.txt")
RESULTS_DIR = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")
BEST_MODEL  = "3.3B-blocks-arn-es-cascade_results.json"
OUT_FILE    = Path("/home/it238/projects/mapudungun/mapudungun-mt/notes/codeswitching_analysis.txt")


def es_fraction(model, sentence: str) -> tuple[float, list[str]]:
    words = re.findall(r"\b\w+\b", sentence, flags=re.UNICODE)
    long_words = [w for w in words if len(w) >= 3]
    if not long_words:
        return 0.0, []
    es_words = []
    for w in long_words:
        labels, _ = model.predict(w.lower(), k=1)
        if labels and labels[0] == "__label__es":
            es_words.append(w)
    return len(es_words) / len(long_words), es_words


def main():
    print("Loading fastText LID model...")
    lid = fasttext.load_model(LID_MODEL)

    src_lines = TEST_SRC.read_text().splitlines()
    tgt_lines = TEST_TGT.read_text().splitlines()

    results = json.load(open(RESULTS_DIR / BEST_MODEL))
    chrf_scores = [r["chrf"] for r in results]
    pred_lines  = [r["pred"] for r in results]

    assert len(src_lines) == len(chrf_scores), \
        f"Length mismatch: src={len(src_lines)}, results={len(chrf_scores)}"
    tgt_lines = [r["ref"] for r in results]  # use ref from results to ensure alignment

    print(f"Analysing {len(src_lines):,} test sentences...")
    es_fracs = []
    es_word_lists = []
    for sent in src_lines:
        frac, es_words = es_fraction(lid, sent)
        es_fracs.append(frac)
        es_word_lists.append(es_words)

    es_fracs = np.array(es_fracs)
    chrf_arr = np.array(chrf_scores)

    # ── summary stats ──────────────────────────────────────────────────────
    any_es   = np.sum(es_fracs > 0)
    heavy_es = np.sum(es_fracs > 0.2)
    pct_any  = 100 * any_es / len(es_fracs)
    pct_heavy = 100 * heavy_es / len(es_fracs)

    pearson_r, pearson_p   = stats.pearsonr(es_fracs, chrf_arr)
    spearman_r, spearman_p = stats.spearmanr(es_fracs, chrf_arr)

    # ── bucket analysis ────────────────────────────────────────────────────
    low_mask  = es_fracs == 0
    med_mask  = (es_fracs > 0) & (es_fracs <= 0.2)
    high_mask = es_fracs > 0.2

    buckets = [
        ("No CS (0%)",       low_mask),
        ("Light CS (1–20%)", med_mask),
        ("Heavy CS (>20%)",  high_mask),
    ]

    lines = []
    lines.append("=" * 70)
    lines.append("CODE-SWITCHING ANALYSIS  |  3.3B cascade  |  arn→es test set")
    lines.append("=" * 70)
    lines.append(f"\nTest sentences:                {len(src_lines):,}")
    lines.append(f"Sentences with ≥1 Spanish word: {any_es:,}  ({pct_any:.1f}%)")
    lines.append(f"Sentences >20% Spanish words:   {heavy_es:,}  ({pct_heavy:.1f}%)")
    lines.append(f"\nCorrelation (Spanish word% vs chrF++):")
    lines.append(f"  Pearson  r = {pearson_r:+.3f}  (p={pearson_p:.2e})")
    lines.append(f"  Spearman r = {spearman_r:+.3f}  (p={spearman_p:.2e})")
    lines.append(f"\nMean chrF++ by code-switching level:")
    for label, mask in buckets:
        if mask.sum() > 0:
            lines.append(f"  {label:<22}  n={mask.sum():5,}  mean chrF++ = {chrf_arr[mask].mean():.2f}")

    # ── examples: high CS, best predictions ───────────────────────────────
    lines.append("\n" + "-" * 70)
    lines.append("HIGH CS SENTENCES — best predictions (chrF++ ≥ 50)")
    lines.append("-" * 70)
    high_good = [(chrf_arr[i], src_lines[i], tgt_lines[i], pred_lines[i], es_word_lists[i])
                 for i in range(len(src_lines)) if high_mask[i] and chrf_arr[i] >= 50]
    high_good.sort(reverse=True)
    for chrf, src, ref, pred, es_words in high_good[:8]:
        lines.append(f"\n  chrF={chrf:.1f}  ES-words: {es_words}")
        lines.append(f"  SRC:  {src}")
        lines.append(f"  REF:  {ref}")
        lines.append(f"  PRED: {pred}")

    # ── examples: high CS, worst predictions ──────────────────────────────
    lines.append("\n" + "-" * 70)
    lines.append("HIGH CS SENTENCES — worst predictions (chrF++ < 20)")
    lines.append("-" * 70)
    high_bad = [(chrf_arr[i], src_lines[i], tgt_lines[i], pred_lines[i], es_word_lists[i])
                for i in range(len(src_lines)) if high_mask[i] and chrf_arr[i] < 20]
    high_bad.sort()
    for chrf, src, ref, pred, es_words in high_bad[:8]:
        lines.append(f"\n  chrF={chrf:.1f}  ES-words: {es_words}")
        lines.append(f"  SRC:  {src}")
        lines.append(f"  REF:  {ref}")
        lines.append(f"  PRED: {pred}")

    # ── examples: no CS, worst predictions ────────────────────────────────
    lines.append("\n" + "-" * 70)
    lines.append("NO CS SENTENCES — worst predictions (chrF++ < 10)")
    lines.append("-" * 70)
    low_bad = [(chrf_arr[i], src_lines[i], tgt_lines[i], pred_lines[i])
               for i in range(len(src_lines)) if low_mask[i] and chrf_arr[i] < 10]
    low_bad.sort()
    for chrf, src, ref, pred in low_bad[:8]:
        lines.append(f"\n  chrF={chrf:.1f}")
        lines.append(f"  SRC:  {src}")
        lines.append(f"  REF:  {ref}")
        lines.append(f"  PRED: {pred}")

    output = "\n".join(lines)
    print(output)
    OUT_FILE.write_text(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
