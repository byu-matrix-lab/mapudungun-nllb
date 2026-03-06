#!/usr/bin/env python3
"""
Word-level language ID on the Mapudungun training corpus using fastText lid.176.
Reports what fraction of tokens are identified as Spanish (es) vs other,
as a better measure of code-switching than function-word matching.
"""

import re
import random
from collections import Counter
from pathlib import Path

import fasttext

LID_MODEL = "/home/it238/nobackup/autodelete/mapudungun/lid.176.bin"
DATA_ROOT = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed")

APPROACHES = ["blocks", "lines"]


def predict_word(model, word):
    labels, _ = model.predict(word.lower(), k=1)
    if labels:
        return labels[0].replace("__label__", "")
    return "unk"


def analyze_file(model, src_path, sample_lines=10000):
    lines = src_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > sample_lines:
        random.seed(42)
        lines = random.sample(lines, sample_lines)

    word_lang_counts = Counter()
    lines_with_es = 0
    total_lines = 0
    es_word_fracs = []

    for line in lines:
        words = re.findall(r"\b\w+\b", line, flags=re.UNICODE)
        if not words:
            continue
        total_lines += 1
        line_es = 0
        for w in words:
            if len(w) < 3:  # skip very short tokens — unreliable
                continue
            lang = predict_word(model, w)
            word_lang_counts[lang] += 1
            if lang == "es":
                line_es += 1
        frac = line_es / len(words) if words else 0.0
        es_word_fracs.append(frac)
        if frac > 0.2:
            lines_with_es += 1

    return word_lang_counts, lines_with_es, total_lines, es_word_fracs, lines


def main():
    print(f"Loading fastText LID model from {LID_MODEL} ...")
    model = fasttext.load_model(LID_MODEL)
    print("Model loaded.\n")

    for approach in APPROACHES:
        src_path = DATA_ROOT / approach / "cleaned" / "train" / "cleaned" / "src.txt"
        if not src_path.exists():
            print(f"[SKIP] {src_path} not found")
            continue

        print(f"{'='*60}")
        print(f"Approach: {approach}  |  {src_path}")
        print(f"{'='*60}")
        counts, lines_es, total, fracs, sampled_lines = analyze_file(model, src_path)

        total_words = sum(counts.values())
        print(f"Total words analysed (>=3 chars): {total_words:,}")
        print(f"Total lines sampled:              {total:,}")
        print()
        print("Top-10 predicted languages (by word count):")
        for lang, n in counts.most_common(10):
            pct = 100 * n / total_words if total_words else 0
            print(f"  {lang:8s}  {n:8,}  ({pct:.1f}%)")

        es_pct = 100 * counts.get("es", 0) / total_words if total_words else 0
        lines_es_pct = 100 * lines_es / total if total else 0
        avg_es_frac = 100 * (sum(fracs) / len(fracs)) if fracs else 0
        print()
        print(f"Spanish words:                 {es_pct:.1f}%")
        print(f"Lines >20% Spanish words:      {lines_es_pct:.1f}%")
        print(f"Avg Spanish word% per line:    {avg_es_frac:.1f}%")
        print()

        # Show a few high-CS examples from sampled lines
        high_cs = []
        for line in sampled_lines:
            words = re.findall(r"\b\w+\b", line, flags=re.UNICODE)
            if len(words) < 4:
                continue
            long_words = [w for w in words if len(w) >= 3]
            if not long_words:
                continue
            es_count = sum(1 for w in long_words if predict_word(model, w) == "es")
            frac = es_count / len(long_words)
            if frac > 0.5:
                high_cs.append((frac, line))
        high_cs.sort(reverse=True)
        print(f"Example lines >50% Spanish words (up to 5):")
        for frac, line in high_cs[:5]:
            print(f"  [{frac:.0%} es]  {line[:120]}")
        print()


if __name__ == "__main__":
    main()
