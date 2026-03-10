#!/usr/bin/env python3
"""
comet_score.py

Score all prediction files with COMET (wmt22-comet-da).
Reads prediction .txt files alongside their corresponding .json results files,
adds comet score to the JSON, and prints a summary table.

Usage:
  python scripts/eval/comet_score.py \
      --results-dir /home/it238/nobackup/autodelete/mapudungun/results \
      --data-dir    /home/it238/nobackup/autodelete/mapudungun/data-processed \
      --comet-model Unbabel/wmt22-comet-da
"""

import sys
import types

_pkg = types.ModuleType("pkg_resources")
_pkg.DistributionNotFound = Exception
_pkg.get_distribution = lambda name: None
sys.modules.setdefault("pkg_resources", _pkg)

import argparse
import json
import logging
from pathlib import Path

from comet import download_model, load_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    COMET_CKPT = (
        "/home/it238/groups/grp_ladle/nobackup/autodelete/comet_model/"
        "models--Unbabel--wmt22-comet-da/snapshots/"
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    parser.add_argument("--comet-model", default=COMET_CKPT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()



def find_prediction_files(results_dir):
    """Find all *_preds.txt files and their paired _results.json files."""
    results_dir = Path(results_dir)
    pairs = []
    for pred_file in sorted(results_dir.rglob("*_preds.txt")):
        json_file = pred_file.with_name(pred_file.name.replace("_preds.txt", "_results.json"))
        if json_file.exists():
            pairs.append((json_file, pred_file))
    return pairs


def main():
    args = parse_args()

    logger.info(f"Loading COMET model from: {args.comet_model}")
    # If it's a local .ckpt path, load directly; otherwise download from HF
    if args.comet_model.endswith(".ckpt"):
        model_path = args.comet_model
    else:
        model_path = download_model(args.comet_model)
    comet_model = load_from_checkpoint(model_path)

    pairs = find_prediction_files(args.results_dir)
    logger.info(f"Found {len(pairs)} prediction files to score")

    all_results = []

    # Derive a summary JSON path alongside each results file
    for json_file, pred_file in pairs:
        run_name = pred_file.stem.replace("_preds", "")
        summary_file = pred_file.parent / f"{run_name}_comet.json"

        # Skip if already scored
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            logger.info(f"Already scored: {run_name} (comet={summary['comet']})")
            all_results.append(summary)
            continue

        # Load per-sentence results written by predict_test.py (list of dicts)
        sentences = json.loads(json_file.read_text())
        src_lines = [s["src"] for s in sentences]
        ref_lines  = [s["ref"] for s in sentences]
        predictions = [s["pred"] for s in sentences]

        data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(src_lines, predictions, ref_lines)
        ]

        logger.info(f"Scoring {run_name} ({len(data):,} sentences)...")
        output = comet_model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
        comet_score = round(output.system_score, 4)

        corpus_chrf = round(
            sum(s.get("chrf", 0) for s in sentences) / max(len(sentences), 1), 2
        )
        summary = {"run": run_name, "n": len(sentences), "chrf": corpus_chrf, "comet": comet_score}
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"  COMET = {comet_score}  chrF = {corpus_chrf}")
        all_results.append(summary)

    # Summary table
    print("\n" + "="*60)
    print(f"{'Run':<45} {'chrF':>6} {'COMET':>7}")
    print("="*60)
    for m in sorted(all_results, key=lambda x: x.get("run", "")):
        print(f"{m.get('run','?'):<45} {m.get('chrf',0):>6.2f} {m.get('comet',0):>7.4f}")


if __name__ == "__main__":
    main()
