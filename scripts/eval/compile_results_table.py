#!/usr/bin/env python3
"""
compile_results_table.py

Produce a unified BLEU / chrF++ / COMET table for all prediction files.

Usage:
  python scripts/eval/compile_results_table.py
  python scripts/eval/compile_results_table.py --results-dir /path/to/predictions --out results/metrics_table.tsv
"""

import argparse
import json
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import CHRF

RESULTS_DIR = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")
DATA_DIR    = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed")
ARN_REF = str(DATA_DIR / "blocks/cleaned/test/cleaned/src.txt")
ES_REF  = str(DATA_DIR / "blocks/cleaned/test/cleaned/tgt.txt")

# Old buggy files to skip (stale arn→es weights)
SKIP_RUNS = {"arn-es", "es-arn"}


def infer_ref(run_name: str) -> str | None:
    """Return path to reference file based on translation direction in run name."""
    if "-arn-es" in run_name or run_name.endswith("arn_es"):
        return ES_REF
    if "-es-arn" in run_name or run_name.endswith("es_arn"):
        return ARN_REF
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--out", default=None, help="Write TSV to this path (default: print only)")
    p.add_argument("--skip-missing-comet", action="store_true",
                   help="Skip rows where comet.json doesn't exist yet")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    chrf_metric = CHRF(word_order=2)

    rows = []
    for pred_file in sorted(results_dir.rglob("*_preds.txt")):
        run_name = pred_file.stem.replace("_preds", "")
        if run_name in SKIP_RUNS:
            continue

        ref_path = infer_ref(run_name)
        if ref_path is None:
            print(f"WARN: cannot infer direction for {run_name}, skipping")
            continue

        results_json = pred_file.with_name(f"{run_name}_results.json")
        comet_json   = pred_file.with_name(f"{run_name}_comet.json")

        if args.skip_missing_comet and not comet_json.exists():
            continue

        preds = pred_file.read_text(encoding="utf-8").splitlines()
        refs  = Path(ref_path).read_text(encoding="utf-8").splitlines()

        if len(preds) != len(refs):
            print(f"WARN: length mismatch for {run_name} ({len(preds)} preds vs {len(refs)} refs), skipping")
            continue

        chrf_score = chrf_metric.corpus_score(preds, [refs]).score
        bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score

        comet_score = None
        if comet_json.exists():
            comet_score = json.loads(comet_json.read_text()).get("comet")

        rows.append({
            "run": run_name,
            "chrf": round(chrf_score, 2),
            "bleu": round(bleu_score, 2),
            "comet": round(comet_score, 4) if comet_score is not None else "—",
        })

    rows.sort(key=lambda r: r["run"])

    header = f"{'Run':<50} {'chrF++':>7} {'BLEU':>7} {'COMET':>7}"
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        comet_str = f"{r['comet']:>7.4f}" if isinstance(r["comet"], float) else f"{'—':>7}"
        print(f"{r['run']:<50} {r['chrf']:>7.2f} {r['bleu']:>7.2f} {comet_str}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("run\tchrf\tbleu\tcomet\n")
            for r in rows:
                f.write(f"{r['run']}\t{r['chrf']}\t{r['bleu']}\t{r['comet']}\n")
        print(f"\nTable written to {out}")


if __name__ == "__main__":
    main()
