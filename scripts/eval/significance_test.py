#!/usr/bin/env python3
"""
Bootstrap significance testing for chrF++ differences between systems.

Compares pairs of interest using paired bootstrap resampling (n=10000).
Reports p-value, effect size (delta chrF++), and 95% CI.

Usage:
  uv run python scripts/eval/significance_test.py
"""

import json
import random
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")
N_BOOTSTRAP = 10_000
SEED = 42


def load_scores(run_name: str) -> list[float]:
    path = RESULTS_DIR / f"{run_name}_results.json"
    return [r["chrf"] for r in json.load(open(path))]


def bootstrap_test(scores_a: list[float], scores_b: list[float],
                   n: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Paired bootstrap: is mean(B) - mean(A) significant?"""
    rng = random.Random(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Score lists must be same length"

    observed_delta = b.mean() - a.mean()
    deltas = []
    idx = list(range(len(a)))
    for _ in range(n):
        sample = rng.choices(idx, k=len(idx))
        deltas.append(b[sample].mean() - a[sample].mean())

    deltas = np.array(deltas)
    # Shift bootstrap distribution to null (H0: delta=0), then compute two-tailed p
    deltas_shifted = deltas - observed_delta
    p = np.mean(np.abs(deltas_shifted) >= np.abs(observed_delta))
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    return {
        "delta": observed_delta,
        "p": p,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": p < 0.05,
    }


def fmt(name_a, name_b, result):
    sig = "✓" if result["significant"] else "✗"
    return (f"  {sig}  {name_b} vs {name_a:<45}"
            f"  Δ={result['delta']:+.2f}"
            f"  95%CI=[{result['ci_lo']:+.2f},{result['ci_hi']:+.2f}]"
            f"  p={result['p']:.4f}")


def main():
    # Pairs of interest: (baseline, system, label_a, label_b)
    comparisons = [
        # ── arn→es ────────────────────────────────────────────────────────
        ("3.3B-arn-es",                    "3.3B-blocks-arn-es-morfessor_vc",
         "3.3B Standard BPE",              "3.3B Morfessor-VC"),
        ("3.3B-arn-es-morfessor",          "3.3B-blocks-arn-es-morfessor_vc",
         "3.3B Morfessor",                 "3.3B Morfessor-VC"),
        ("1.3B-arn-es-morfessor",          "1.3B-blocks-arn-es-morfessor_vc",
         "1.3B Morfessor",                 "1.3B Morfessor-VC"),
        ("3.3B-arn-es",                    "3.3B-arn-es-morfessor",
         "3.3B Standard BPE",              "3.3B Morfessor"),
        ("600M-arn-es-morfessor",          "3.3B-arn-es",
         "600M Morfessor",                 "3.3B Standard BPE"),
        ("600M-blocks-arn-es-morfessor_vc",     "3.3B-arn-es",
         "600M Morfessor-VC",              "3.3B Standard BPE"),
        ("3.3B-blocks-arn-es-joint_bpe",    "3.3B-blocks-arn-es-morfessor_vc",
         "3.3B Joint-5K BPE",              "3.3B Morfessor-VC"),
        ("3.3B-blocks-arn-es-mono_bpe",   "3.3B-blocks-arn-es-morfessor_vc",
         "3.3B Mono BPE",                  "3.3B Morfessor-VC"),
        # ── es→arn ────────────────────────────────────────────────────────
        ("3.3B-es-arn",                    "3.3B-blocks-es-arn-morfessor_vc",
         "3.3B Standard BPE",              "3.3B Morfessor-VC"),
        ("3.3B-es-arn",                    "3.3B-es-arn-morfessor",
         "3.3B Standard BPE",              "3.3B Morfessor"),
        ("1.3B-es-arn",                    "1.3B-blocks-es-arn-morfessor_vc",
         "1.3B Standard BPE",              "1.3B Morfessor-VC"),
    ]

    print(f"\nPaired bootstrap significance test  (n={N_BOOTSTRAP:,} resamples)")
    print("=" * 100)

    print("\n── arn→es ─────────────────────────────────────────────────────────────────────")
    arn_es_pairs = comparisons[:8]
    for run_a, run_b, label_a, label_b in arn_es_pairs:
        try:
            scores_a = load_scores(run_a)
            scores_b = load_scores(run_b)
            r = bootstrap_test(scores_a, scores_b)
            print(fmt(label_a, label_b, r))
        except FileNotFoundError as e:
            print(f"  SKIP  {e}")

    print("\n── es→arn ─────────────────────────────────────────────────────────────────────")
    for run_a, run_b, label_a, label_b in comparisons[8:]:
        try:
            scores_a = load_scores(run_a)
            scores_b = load_scores(run_b)
            r = bootstrap_test(scores_a, scores_b)
            print(fmt(label_a, label_b, r))
        except FileNotFoundError as e:
            print(f"  SKIP  {e}")

    print()


if __name__ == "__main__":
    main()
