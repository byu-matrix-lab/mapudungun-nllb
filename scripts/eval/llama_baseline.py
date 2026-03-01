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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sacrebleu

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
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
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


def load_split(data_dir, approach, split):
    base = Path(data_dir) / approach / "cleaned" / split / "cleaned"
    src = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    return list(zip(src, tgt))


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    dev_pairs = load_split(args.data_dir, args.approach, "dev")
    test_pairs = load_split(args.data_dir, args.approach, "test")
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

    chrf = sacrebleu.corpus_chrf(predictions, [test_tgt])
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

    stem = f"llama_{args.approach}_{args.src}_{args.tgt}"
    (output_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    (output_dir / f"{stem}_predictions.txt").write_text(
        "\n".join(predictions), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
