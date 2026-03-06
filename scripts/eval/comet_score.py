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
    parser.add_argument("--data-dir", required=True)
    COMET_CKPT = (
        "/home/it238/groups/grp_ladle/nobackup/autodelete/comet_model/"
        "models--Unbabel--wmt22-comet-da/snapshots/"
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    parser.add_argument("--comet-model", default=COMET_CKPT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()


def load_test_refs(data_dir, approach, src, tgt):
    """Load source and reference (target) test sentences."""
    base = Path(data_dir) / approach / "cleaned" / "test" / "cleaned"
    src_lines = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt_lines = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    return src_lines, tgt_lines


def find_prediction_files(results_dir):
    """Find all *_predictions.txt files and their paired .json files."""
    results_dir = Path(results_dir)
    pairs = []
    for pred_file in sorted(results_dir.rglob("*_predictions.txt")):
        json_file = pred_file.with_name(pred_file.name.replace("_predictions.txt", ".json"))
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

    for json_file, pred_file in pairs:
        meta = json.loads(json_file.read_text())

        # Skip if already scored
        if "comet" in meta:
            logger.info(f"Already scored: {json_file.name} (comet={meta['comet']})")
            all_results.append(meta)
            continue

        approach = meta.get("approach", "lines")
        src = meta.get("src", "arn")
        tgt = meta.get("tgt", "es")

        try:
            src_lines, ref_lines = load_test_refs(args.data_dir, approach, src, tgt)
        except Exception as e:
            logger.warning(f"Could not load refs for {json_file.name}: {e}")
            continue

        predictions = pred_file.read_text(encoding="utf-8").splitlines()

        if len(predictions) != len(src_lines):
            logger.warning(
                f"{pred_file.name}: {len(predictions)} preds vs {len(src_lines)} refs — skipping"
            )
            continue

        data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(src_lines, predictions, ref_lines)
        ]

        logger.info(f"Scoring {json_file.name} ({len(data):,} sentences)...")
        output = comet_model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
        comet_score = round(output.system_score, 4)

        meta["comet"] = comet_score
        json_file.write_text(json.dumps(meta, indent=2))
        logger.info(f"  COMET = {comet_score}")
        all_results.append(meta)

    # Summary table
    print("\n" + "="*80)
    print(f"{'File':<55} {'chrF':>6} {'BLEU':>6} {'COMET':>7}")
    print("="*80)
    for m in sorted(all_results, key=lambda x: (x.get("src",""), x.get("approach",""), str(x.get("model","")))):
        name = f"{m.get('model','?').split('/')[-1]} {m.get('approach','?')} {m.get('src','?')}→{m.get('tgt','?')}"
        print(f"{name:<55} {m.get('chrf',0):>6.2f} {m.get('bleu',0):>6.2f} {m.get('comet',0):>7.4f}")


if __name__ == "__main__":
    main()
