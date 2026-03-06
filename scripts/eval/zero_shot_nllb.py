#!/usr/bin/env python3
"""
zero_shot_nllb.py

Evaluate the base NLLB-200 model (no fine-tuning) on the Mapudungun–Spanish
test set. Provides a baseline to quantify how much fine-tuning helps.

Usage:
  python scripts/eval/zero_shot_nllb.py \\
      --approach lines --src arn --tgt es \\
      --data-dir /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --output-dir /home/it238/nobackup/autodelete/mapudungun/results/zero_shot
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_CODES = {"arn": "arn_Latn", "es": "spa_Latn"}
PROXY_LANGS = ["ayr_Latn", "grn_Latn", "quy_Latn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot NLLB-200 baseline.")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--approach", choices=["lines", "blocks"], required=True)
    parser.add_argument("--src", choices=["arn", "es"], required=True)
    parser.add_argument("--tgt", choices=["arn", "es"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


def load_test(data_dir, approach):
    base = Path(data_dir) / approach / "cleaned" / "test" / "cleaned"
    src = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(tgt)
    return src, tgt


def add_arn_latn_token(tokenizer, model):
    if "arn_Latn" in tokenizer.get_vocab():
        logger.info("arn_Latn already in vocabulary.")
        return
    tokenizer.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    model.resize_token_embeddings(len(tokenizer))
    proxy_ids = [tokenizer.convert_tokens_to_ids(l) for l in PROXY_LANGS]
    new_id = tokenizer.convert_tokens_to_ids("arn_Latn")
    with torch.no_grad():
        shared = model.model.shared.weight
        shared[new_id] = shared[proxy_ids].mean(dim=0)
    logger.info(f"arn_Latn (id={new_id}) initialized from proxy mean.")


def main():
    args = parse_args()
    assert args.src != args.tgt
    src_code = LANG_CODES[args.src]
    tgt_code = LANG_CODES[args.tgt]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model     : {args.model}")
    logger.info(f"Direction : {args.src} ({src_code}) → {args.tgt} ({tgt_code})")
    logger.info(f"Approach  : {args.approach}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    add_arn_latn_token(tokenizer, model)
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    model.generation_config.forced_bos_token_id = tgt_token_id

    src_lines, tgt_lines = load_test(args.data_dir, args.approach)
    logger.info(f"Test set: {len(src_lines):,} pairs")

    tokenizer.src_lang = src_code
    predictions = []
    for i in range(0, len(src_lines), args.batch_size):
        batch = src_lines[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                num_beams=args.num_beams,
                max_length=args.max_length,
            )
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        if (i // args.batch_size) % 100 == 0:
            logger.info(f"  {i:,} / {len(src_lines):,}")

    chrf = sacrebleu.corpus_chrf(predictions, [tgt_lines])
    bleu = sacrebleu.corpus_bleu(predictions, [tgt_lines])
    results = {
        "model": args.model,
        "approach": args.approach,
        "src": args.src,
        "tgt": args.tgt,
        "n_pairs": len(src_lines),
        "chrf": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
    }
    logger.info(f"Results: {results}")

    model_tag = args.model.split("/")[-1].replace("nllb-200-distilled-", "nllb-")
    stem = f"zero_shot_{model_tag}_{args.approach}_{args.src}_{args.tgt}"
    (output_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    (output_dir / f"{stem}_predictions.txt").write_text(
        "\n".join(predictions), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
