"""
analyze_corpus.py

Character and corpus statistics analysis for extracted parallel text.
Helps inform data cleaning configuration (arn-CL.yaml) by showing:
  - Character frequencies and Unicode code points (flagging rare/unexpected chars)
  - Basic corpus statistics (pair counts, token lengths, length ratios)
  - Duplicate and source==target counts

Usage:
  python scripts/data/analyze_corpus.py \
      --src /path/to/train.arn \
      --tgt /path/to/train.es \
      --top 50 \
      --rare-threshold 0.0001
"""

import argparse
import unicodedata
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Character and corpus statistics for a parallel text file pair."
    )
    parser.add_argument("--src", required=True, help="Source language file (.arn)")
    parser.add_argument("--tgt", required=True, help="Target language file (.es)")
    parser.add_argument(
        "--top", type=int, default=40,
        help="Number of most-frequent non-ASCII characters to show (default: 40)."
    )
    parser.add_argument(
        "--rare-threshold", type=float, default=0.0001,
        help="Characters below this fraction of total chars are flagged as rare (default: 0.0001)."
    )
    return parser.parse_args()


def load_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def char_report(lines: list[str], label: str, top: int, rare_threshold: float):
    all_text = "\n".join(lines)
    total_chars = len(all_text)
    counts = Counter(all_text)

    ascii_chars = {ch: n for ch, n in counts.items() if ord(ch) < 128}
    non_ascii = {ch: n for ch, n in counts.items() if ord(ch) >= 128}

    print(f"\n{'='*60}")
    print(f"  CHARACTER ANALYSIS — {label}")
    print(f"{'='*60}")
    print(f"  Total characters : {total_chars:,}")
    print(f"  Unique characters: {len(counts):,}  "
          f"(ASCII: {len(ascii_chars)}, non-ASCII: {len(non_ascii)})")

    # Non-ASCII characters sorted by frequency
    print(f"\n  Non-ASCII characters (sorted by frequency):")
    print(f"  {'Char':<6} {'Decimal':<9} {'Hex':<8} {'Unicode name':<40} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*80}")
    for ch, n in sorted(non_ascii.items(), key=lambda x: -x[1])[:top]:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "(no name)"
        pct = n / total_chars * 100
        print(f"  {ch!r:<6} {ord(ch):<9} U+{ord(ch):04X}  {name:<40} {n:>8,}  {pct:>5.3f}%")

    # Rare characters (potential noise)
    rare = {ch: n for ch, n in counts.items()
            if n / total_chars < rare_threshold and ord(ch) >= 32}
    if rare:
        print(f"\n  Rare characters (< {rare_threshold*100:.4f}% of total):")
        print(f"  {'Char':<6} {'Decimal':<9} {'Hex':<8} {'Unicode name':<40} {'Count':>8}")
        print(f"  {'-'*76}")
        for ch, n in sorted(rare.items(), key=lambda x: x[1]):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "(no name)"
            print(f"  {ch!r:<6} {ord(ch):<9} U+{ord(ch):04X}  {name:<40} {n:>8,}")
    else:
        print(f"\n  No rare characters below threshold {rare_threshold}.")

    # ASCII punctuation and symbols (non-alphanumeric, non-space)
    ascii_punct = {ch: n for ch, n in ascii_chars.items()
                   if not ch.isalnum() and ch != ' '}
    if ascii_punct:
        print(f"\n  ASCII punctuation/symbols present:")
        print(f"  " + "  ".join(
            f"{ch!r}({n:,})" for ch, n in sorted(ascii_punct.items(), key=lambda x: -x[1])
        ))


def corpus_stats(src_lines: list[str], tgt_lines: list[str]):
    print(f"\n{'='*60}")
    print(f"  CORPUS STATISTICS")
    print(f"{'='*60}")

    n = len(src_lines)
    print(f"  Total pairs: {n:,}")

    # Token lengths (whitespace-split)
    src_lens = [len(l.split()) for l in src_lines]
    tgt_lens = [len(l.split()) for l in tgt_lines]

    def stats(vals, label):
        if not vals:
            return
        avg = sum(vals) / len(vals)
        srt = sorted(vals)
        p5  = srt[int(len(srt) * 0.05)]
        p25 = srt[int(len(srt) * 0.25)]
        p50 = srt[int(len(srt) * 0.50)]
        p75 = srt[int(len(srt) * 0.75)]
        p95 = srt[int(len(srt) * 0.95)]
        p99 = srt[int(len(srt) * 0.99)]
        print(f"\n  {label} token length (whitespace-split words):")
        print(f"    mean={avg:.1f}  min={min(vals)}  max={max(vals)}")
        print(f"    p5={p5}  p25={p25}  p50={p50}  p75={p75}  p95={p95}  p99={p99}")
        empty = sum(1 for v in vals if v == 0)
        if empty:
            print(f"    Empty lines: {empty:,}")

    stats(src_lens, "Source (arn)")
    stats(tgt_lens, "Target (es)")

    # Length ratios (src_words / tgt_words)
    ratios = []
    for s, t in zip(src_lens, tgt_lens):
        if t > 0 and s > 0:
            ratios.append(s / t)
    if ratios:
        avg_r = sum(ratios) / len(ratios)
        extreme = sum(1 for r in ratios if r > 4 or r < 0.25)
        print(f"\n  Length ratios (src_words / tgt_words):")
        print(f"    mean={avg_r:.2f}  min={min(ratios):.2f}  max={max(ratios):.2f}")
        print(f"    Pairs with ratio > 4 or < 0.25: {extreme:,}  ({extreme/n*100:.1f}%)")

    # Duplicates
    src_set = Counter(src_lines)
    tgt_set = Counter(tgt_lines)
    pair_set = Counter(zip(src_lines, tgt_lines))
    exact_dups = sum(c - 1 for c in pair_set.values() if c > 1)
    src_eq_tgt = sum(1 for s, t in zip(src_lines, tgt_lines) if s.strip() == t.strip())
    print(f"\n  Duplicate pairs (exact): {exact_dups:,}  ({exact_dups/n*100:.1f}%)")
    print(f"  Source == target:        {src_eq_tgt:,}  ({src_eq_tgt/n*100:.1f}%)")

    # Very short pairs
    for threshold in [1, 2, 3]:
        short = sum(1 for s, t in zip(src_lens, tgt_lens)
                    if s <= threshold or t <= threshold)
        print(f"  Pairs with src or tgt <= {threshold} word(s): {short:,}  ({short/n*100:.1f}%)")

    # Residual tag patterns
    import re
    tag_patterns = {
        "<*...> tags (stripped)": r"<\*\w+>",
        "Incomplete tags (e.g. SPA>)": r"(?<![<\w])\w+>",
        "-/.../-  +/.../+ patterns": r"[-+]/[^/]+/[-+]",
        "% symbols": r"%",
        "@ symbols": r"@",
    }
    print(f"\n  Residual noise patterns in source (arn):")
    src_text = "\n".join(src_lines)
    for desc, pat in tag_patterns.items():
        matches = re.findall(pat, src_text)
        if matches:
            examples = list(dict.fromkeys(matches))[:5]
            print(f"    {desc}: {len(matches):,} occurrences  e.g. {examples}")
        else:
            print(f"    {desc}: 0")


def main():
    args = parse_args()
    src_path = Path(args.src)
    tgt_path = Path(args.tgt)

    print(f"Analyzing: {src_path.name} + {tgt_path.name}")

    src_lines = load_lines(src_path)
    tgt_lines = load_lines(tgt_path)

    if len(src_lines) != len(tgt_lines):
        print(f"WARNING: line count mismatch — src={len(src_lines):,}, tgt={len(tgt_lines):,}")

    corpus_stats(src_lines, tgt_lines)
    char_report(src_lines, f"Source — {src_path.name}", args.top, args.rare_threshold)
    char_report(tgt_lines, f"Target — {tgt_path.name}", args.top, args.rare_threshold)

    print()


if __name__ == "__main__":
    main()
