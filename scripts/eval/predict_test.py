#!/usr/bin/env python3
"""
Run fine-tuned NLLB model on the full test set and save predictions.
Computes per-sentence chrF++ and reports the distribution.
Usage:
  python predict_test.py --direction arn-es [--model-path /path/to/model]
  python predict_test.py --direction es-arn [--model-path /path/to/model]
  python predict_test.py --direction arn-es --morfessor [--model-path /path/to/morfessor/model]
"""

import argparse
import re
import json
import torch
from pathlib import Path
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import CHRF

BASE_TOKENIZER = "/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
MORFESSOR_MODEL = "/home/it238/nobackup/autodelete/mapudungun/morfessor_model.bin"
DEFAULT_MODELS = {
    "arn-es": "/home/it238/nobackup/autodelete/mapudungun/models/nllb-200-3.3B-blocks-arn-es/best",
    "es-arn": "/home/it238/nobackup/autodelete/mapudungun/models/nllb-200-3.3B-blocks-es-arn/best",
}
# src = text fed to model; ref = reference for scoring
TEST_DATA = {
    "arn-es": {
        "src": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/src.txt",
        "ref": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/tgt.txt",
        "src_lang": "arn_Latn",
        "tgt_lang": "spa_Latn",
    },
    "es-arn": {
        "src": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/tgt.txt",
        "ref": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/src.txt",
        "src_lang": "spa_Latn",
        "tgt_lang": "arn_Latn",
    },
    # Morfessor arn→es: segment the Mapudungun test inputs before feeding to model.
    # Score generated Spanish against Spanish references (unaffected by segmentation).
    "arn-es-morfessor": {
        "src": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/morfessor/test/src.txt",
        "ref": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/tgt.txt",
        "src_lang": "arn_Latn",
        "tgt_lang": "spa_Latn",
        "desegment_output": False,
    },
    # Morfessor es→arn: feed clean Spanish, desegment Mapudungun output before scoring.
    "es-arn-morfessor": {
        "src": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/tgt.txt",
        "ref": "/home/it238/nobackup/autodelete/mapudungun/data-processed/blocks/cleaned/test/cleaned/src.txt",
        "src_lang": "spa_Latn",
        "tgt_lang": "arn_Latn",
        "desegment_output": True,
    },
}
OUT_DIR = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")
LANG_CODE_RE = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3,}\s*")
DESEG_RE = re.compile(r"@@ ?")
BATCH_SIZE = 16


def load_tokenizer():
    tok = NllbTokenizer.from_pretrained(BASE_TOKENIZER)
    tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    return tok


def translate_batch(model, tokenizer, texts, src_lang, tgt_id, device):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [LANG_CODE_RE.sub("", s).strip() for s in decoded]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["arn-es", "es-arn"], required=True)
    parser.add_argument("--morfessor", action="store_true",
                        help="Use morfessor test config (segmented arn→es input or desegmented es→arn output)")
    parser.add_argument("--model-path", default=None,
                        help="Override model path (default: hardcoded 3.3B path)")
    parser.add_argument("--run-name", default=None,
                        help="Tag for output filenames, e.g. '1.3B-arn-es-morfessor'. Defaults to direction.")
    args = parser.parse_args()

    cfg_key = f"{args.direction}-morfessor" if args.morfessor else args.direction
    cfg = TEST_DATA[cfg_key]
    desegment_output = cfg.get("desegment_output", False)
    model_path = args.model_path or DEFAULT_MODELS[args.direction]
    run_name = args.run_name or cfg_key

    srcs = Path(cfg["src"]).read_text().splitlines()
    refs = Path(cfg["ref"]).read_text().splitlines()
    assert len(srcs) == len(refs), "src/ref length mismatch"
    print(f"Test set: {len(srcs)} sentences  ({cfg_key})")
    if desegment_output:
        print("Morfessor mode: desegmenting model output before scoring")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading tokenizer...")
    tok = load_tokenizer()
    tgt_id = tok.convert_tokens_to_ids(cfg["tgt_lang"])
    print(f"forced_bos_token_id = {tgt_id} ({cfg['tgt_lang']})")
    print("Loading model...")
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.float16).to(device)
    mdl.eval()

    preds = []
    for i in range(0, len(srcs), BATCH_SIZE):
        batch = srcs[i:i + BATCH_SIZE]
        batch_preds = translate_batch(mdl, tok, batch, cfg["src_lang"], tgt_id, device)
        if desegment_output:
            batch_preds = [DESEG_RE.sub("", p) for p in batch_preds]
        preds.extend(batch_preds)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {i}/{len(srcs)} done...")

    print(f"Translation complete. Computing chrF++...")
    chrf = CHRF(word_order=2)
    corpus_score = chrf.corpus_score(preds, [refs])
    print(f"Corpus chrF++: {corpus_score.score:.2f}")

    # Per-sentence scores
    per_sent = [chrf.sentence_score(p, [r]).score for p, r in zip(preds, refs)]

    # Save predictions
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{run_name}_preds.txt"
    out_path.write_text("\n".join(preds) + "\n")
    print(f"Predictions saved to {out_path}")

    # Save detailed results
    results = []
    for src, ref, pred, score in zip(srcs, refs, preds, per_sent):
        results.append({"src": src, "ref": ref, "pred": pred, "chrf": round(score, 2)})
    results_path = OUT_DIR / f"{run_name}_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Per-sentence results saved to {results_path}")

    # Distribution stats
    scores = [r["chrf"] for r in results]
    buckets = {"0-10": 0, "10-25": 0, "25-50": 0, "50-75": 0, "75-100": 0}
    for s in scores:
        if s < 10:     buckets["0-10"] += 1
        elif s < 25:   buckets["10-25"] += 1
        elif s < 50:   buckets["25-50"] += 1
        elif s < 75:   buckets["50-75"] += 1
        else:          buckets["75-100"] += 1

    print(f"\nchrF++ distribution ({len(scores)} sentences):")
    for bucket, count in buckets.items():
        pct = 100 * count / len(scores)
        print(f"  {bucket:8s}: {count:5d}  ({pct:.1f}%)")

    # Show 5 worst examples
    results_sorted = sorted(results, key=lambda x: x["chrf"])
    print("\n5 worst predictions (lowest chrF++):")
    for r in results_sorted[:5]:
        print(f"  chrF={r['chrf']:.1f}  src: {r['src'][:80]}")
        print(f"         ref: {r['ref'][:80]}")
        print(f"         pred: {r['pred'][:80]}")

    # Show 5 best examples
    print("\n5 best predictions (highest chrF++):")
    for r in results_sorted[-5:]:
        print(f"  chrF={r['chrf']:.1f}  src: {r['src'][:80]}")
        print(f"         ref: {r['ref'][:80]}")
        print(f"         pred: {r['pred'][:80]}")


if __name__ == "__main__":
    main()
