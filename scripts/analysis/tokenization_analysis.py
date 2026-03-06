#!/usr/bin/env python3
"""
tokenization_analysis.py

Analyze how NLLB-200's sentencepiece tokenizer handles Mapudungun vs Spanish.
Computes fertility (tokens/word), vocabulary coverage, and token length distributions.

Usage:
  python scripts/analysis/tokenization_analysis.py \
      --data-dir /home/it238/nobackup/autodelete/mapudungun/data-processed \
      --output-dir /home/it238/nobackup/autodelete/mapudungun/results/analysis
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MODELS = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
]
LANG_NAMES = {"arn": "Mapudungun", "es": "Spanish"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--approach", default="lines", choices=["lines", "blocks"])
    return parser.parse_args()


def load_split(data_dir, approach, split, lang):
    path = Path(data_dir) / approach / "cleaned" / split / "cleaned" / f"{'src' if lang == 'arn' else 'tgt'}.txt"
    # Try both src/tgt and explicit language files
    if not path.exists():
        # For es→arn direction, src is es
        alt = Path(data_dir) / approach / "cleaned" / split / "cleaned" / f"{'tgt' if lang == 'arn' else 'src'}.txt"
        if alt.exists():
            path = alt
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l for l in lines if l.strip()]


def tokenize_words(text):
    """Split on whitespace, strip punctuation for word count."""
    return re.findall(r'\S+', text)


def compute_fertility(tokenizer, texts, src_lang_code):
    """Average tokens per whitespace-delimited word."""
    tokenizer.src_lang = src_lang_code
    total_tokens = 0
    total_words = 0
    token_counts = []
    for text in texts:
        words = tokenize_words(text)
        if not words:
            continue
        encoded = tokenizer(text, add_special_tokens=False)
        n_tokens = len(encoded["input_ids"])
        total_tokens += n_tokens
        total_words += len(words)
        token_counts.append(n_tokens / len(words))

    return {
        "fertility": round(total_tokens / total_words, 3) if total_words else 0,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "n_sentences": len(token_counts),
    }


def vocab_coverage(tokenizer, texts, src_lang_code):
    """What fraction of tokens are single-character (over-segmented)?"""
    tokenizer.src_lang = src_lang_code
    char_tokens = 0
    total_tokens = 0
    unk_tokens = 0
    unk_id = tokenizer.unk_token_id

    for text in texts[:2000]:  # sample for speed
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == unk_id)
        char_tokens += sum(1 for t in tokens if len(t.lstrip("▁")) <= 1)

    return {
        "unk_rate": round(unk_tokens / total_tokens, 4) if total_tokens else 0,
        "single_char_rate": round(char_tokens / total_tokens, 4) if total_tokens else 0,
        "total_tokens_sampled": total_tokens,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train+dev+test for both languages
    splits = ["train", "dev", "test"]

    results = {}

    # Use 600M tokenizer (same sentencepiece vocab as 1.3B distilled)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    for lang, lang_code in [("arn", "arn_Latn"), ("es", "spa_Latn")]:
        print(f"\n=== {LANG_NAMES[lang]} ({lang}) ===")
        all_texts = []
        for split in splits:
            try:
                texts = load_split(args.data_dir, args.approach, split, lang)
                all_texts.extend(texts)
                print(f"  {split}: {len(texts):,} sentences")
            except Exception as e:
                print(f"  {split}: could not load ({e})")

        print(f"  Total: {len(all_texts):,} sentences")

        fertility = compute_fertility(tokenizer, all_texts, lang_code)
        coverage = vocab_coverage(tokenizer, all_texts, lang_code)

        results[lang] = {
            "language": LANG_NAMES[lang],
            "lang_code": lang_code,
            "n_sentences": len(all_texts),
            **fertility,
            **coverage,
        }
        print(f"  Fertility: {fertility['fertility']:.3f} tokens/word")
        print(f"  UNK rate:  {coverage['unk_rate']:.4f}")
        print(f"  Single-char token rate: {coverage['single_char_rate']:.4f}")

    # Token length distribution: how long are Mapudungun words vs Spanish words?
    print("\n=== Word length analysis ===")
    for lang in ["arn", "es"]:
        all_texts = []
        for split in splits:
            try:
                all_texts.extend(load_split(args.data_dir, args.approach, split, lang))
            except Exception:
                pass
        words = [w for t in all_texts for w in tokenize_words(t)]
        lengths = [len(w) for w in words]
        results[lang]["avg_word_chars"] = round(sum(lengths) / len(lengths), 2) if lengths else 0
        results[lang]["median_word_chars"] = sorted(lengths)[len(lengths)//2] if lengths else 0
        print(f"  {LANG_NAMES[lang]}: avg word length = {results[lang]['avg_word_chars']:.1f} chars")

    # Summary comparison
    print("\n=== Summary ===")
    print(f"{'Language':<15} {'Fertility':>10} {'UNK rate':>10} {'Avg word len':>14}")
    for lang in ["arn", "es"]:
        r = results[lang]
        print(f"{r['language']:<15} {r['fertility']:>10.3f} {r['unk_rate']:>10.4f} {r['avg_word_chars']:>14.1f}")

    out_path = output_dir / f"tokenization_{args.approach}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()