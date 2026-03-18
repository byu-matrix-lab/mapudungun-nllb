#!/usr/bin/env python3
"""
llama_baseline.py

Few-shot Llama-3.3-70B-Instruct baseline for Mapudungun–Spanish translation.
Loaded in 4-bit quantization (bitsandbytes) to fit on a single A100 80GB.
Samples --num-shots examples from the dev set as in-context examples.

Usage:
  python scripts/eval/llama_baseline.py \\
      --approach lines --src arn --tgt es \\
      --data-dir /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --output-dir /home/it238/nobackup/autodelete/mapudungun/results/llama
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from sacrebleu.metrics import CHRF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_LABELS = {"arn": "Mapudungun", "es": "Spanish"}

SYSTEM_PROMPTS = {
    ("arn", "es"): (
        "You are a translation assistant specializing in Mapudungun and Spanish. "
        "Translate the given Mapudungun sentence into Spanish. "
        "Output only the translation, nothing else."
    ),
    ("es", "arn"): (
        "You are a translation assistant specializing in Mapudungun and Spanish. "
        "Translate the given Spanish sentence into Mapudungun. "
        "Output only the translation, nothing else."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Llama-3.3-70B few-shot MT baseline.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--approach", choices=["lines", "blocks"], default="lines")
    parser.add_argument("--src", choices=["arn", "es"], required=True)
    parser.add_argument("--tgt", choices=["arn", "es"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


PREDS_DIR = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")


def load_split(data_dir, approach, split, src_lang):
    """Load split. When src_lang='es', swap src.txt↔tgt.txt so we feed Spanish."""
    base = Path(data_dir) / approach / "cleaned" / split / "cleaned"
    arn_lines = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    es_lines  = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    if src_lang == "arn":
        return list(zip(arn_lines, es_lines))   # (arn_src, es_ref)
    else:
        return list(zip(es_lines, arn_lines))   # (es_src, arn_ref)


def build_messages(src_text, examples, src, tgt):
    """Chat-format few-shot prompt for instruct model."""
    src_label = LANG_LABELS[src]
    tgt_label = LANG_LABELS[tgt]
    lines = []
    for ex_src, ex_tgt in examples:
        lines.append(f"{src_label}: {ex_src}")
        lines.append(f"{tgt_label}: {ex_tgt}")
        lines.append("")
    lines.append(f"{src_label}: {src_text}")
    lines.append(f"{tgt_label}:")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[(src, tgt)]},
        {"role": "user", "content": "\n".join(lines)},
    ]


def main():
    args = parse_args()
    assert args.src != args.tgt
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model     : {args.model}")
    logger.info(f"Direction : {args.src} → {args.tgt}, approach={args.approach}")
    logger.info(f"Shots     : {args.num_shots}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    dev_pairs = load_split(args.data_dir, args.approach, "dev", args.src)
    test_pairs = load_split(args.data_dir, args.approach, "test", args.src)
    logger.info(f"Dev: {len(dev_pairs):,}  Test: {len(test_pairs):,}")

    examples = random.sample(dev_pairs, min(args.num_shots, len(dev_pairs)))
    tgt_label = LANG_LABELS[args.tgt]

    test_src = [p[0] for p in test_pairs]
    test_tgt = [p[1] for p in test_pairs]

    predictions = []
    for i in range(0, len(test_src), args.batch_size):
        batch_src = test_src[i : i + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(src_text, examples, args.src, args.tgt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for src_text in batch_src
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out, inp in zip(outputs, inputs["input_ids"]):
            new_tokens = out[len(inp):]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            translation = decoded.split("\n")[0].strip()
            prefix = f"{tgt_label}: "
            if translation.startswith(prefix):
                translation = translation[len(prefix):]
            predictions.append(translation)

        if (i // args.batch_size) % 50 == 0:
            logger.info(f"  {i:,} / {len(test_src):,}")

    chrf_metric = CHRF(word_order=2)
    chrf = chrf_metric.corpus_score(predictions, [test_tgt])
    bleu = sacrebleu.corpus_bleu(predictions, [test_tgt])
    results = {
        "model": args.model,
        "approach": args.approach,
        "src": args.src,
        "tgt": args.tgt,
        "num_shots": args.num_shots,
        "n_pairs": len(test_src),
        "chrf": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
    }
    logger.info(f"Results: {results}")

    model_tag = args.model.split("/")[-1].lower().replace("-instruct", "")
    stem = f"{model_tag}_{args.approach}_{args.src}_{args.tgt}"
    (output_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    (output_dir / f"{stem}_predictions.txt").write_text(
        "\n".join(predictions), encoding="utf-8"
    )

    # Write per-sentence results to predictions dir so comet_score.py picks them up
    per_sent_chrf = [
        chrf_metric.sentence_score(p, [r]).score
        for p, r in zip(predictions, test_tgt)
    ]
    run_name = f"{model_tag}-{args.src}-{args.tgt}"
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    (PREDS_DIR / f"{run_name}_preds.txt").write_text(
        "\n".join(predictions), encoding="utf-8"
    )
    per_sentence = [
        {"src": s, "ref": r, "pred": p, "chrf": round(c, 2)}
        for s, r, p, c in zip(test_src, test_tgt, predictions, per_sent_chrf)
    ]
    (PREDS_DIR / f"{run_name}_results.json").write_text(
        json.dumps(per_sentence, ensure_ascii=False, indent=2)
    )
    logger.info(f"Per-sentence results written to {PREDS_DIR / run_name}_results.json")


if __name__ == "__main__":
    main()
