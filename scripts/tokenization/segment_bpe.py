#!/usr/bin/env python3
"""
segment_bpe.py

Train a SentencePiece BPE model and apply it to the Mapudungun side of the corpus.
Produces two conditions:

  --mode duan   Joint arn+es BPE with 5,000 vocab size, replicating Duan et al. (2020).
                Output: data-processed/{approach}/duan_bpe/{split}/{src,tgt}.txt

  --mode large  Language-specific arn-only BPE. Vocabulary size determined by the
                Gowda & May (2020) 95%-character-coverage heuristic.
                Output: data-processed/{approach}/large_bpe/{split}/{src,tgt}.txt

Only the Mapudungun (arn) side is segmented. Spanish (es) is copied unchanged.
Morpheme boundaries are marked with @@ (Moses convention), same as segment_morfessor.py,
so the existing desegment() logic in finetune_nllb.py and predict_test.py works unchanged.

Note: SentencePiece uses ▁ (U+2581) for word boundaries. This script converts
SentencePiece output to the @@ convention so the training pipeline is uniform.

Usage:
    uv run python scripts/tokenization/segment_bpe.py --mode duan
    uv run python scripts/tokenization/segment_bpe.py --mode large
    uv run python scripts/tokenization/segment_bpe.py --mode duan --approaches blocks lines
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm


DATA_ROOT = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed")
BPE_ROOT  = Path("/home/it238/nobackup/autodelete/mapudungun/bpe_models")
SPLITS    = ["train", "dev", "test"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["duan", "large"], required=True,
                        help="duan: 5K joint BPE (Duan et al. 2020); "
                             "large: arn-only BPE with 95%%-coverage vocab")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--bpe-root", default=str(BPE_ROOT),
                        help="Directory to save trained SentencePiece model files")
    parser.add_argument("--approaches", nargs="+", default=["blocks", "lines"])
    parser.add_argument("--large-vocab", type=int, default=None,
                        help="Override vocabulary size for large mode "
                             "(default: auto from 95%% coverage heuristic)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Gowda & May (2020) 95%-coverage heuristic
# ---------------------------------------------------------------------------

def compute_95_coverage_vocab_size(token_file: Path) -> int:
    """
    Estimate BPE vocabulary size for 95% character coverage following
    Gowda & May (2020, Findings of EMNLP).

    Finds V such that the V most-frequent characters cover 95% of all character
    occurrences, then scales by 10x as a subword vocabulary estimate.
    """
    char_freq: dict[str, int] = {}
    total = 0
    with open(token_file, encoding="utf-8") as f:
        for line in f:
            for ch in line.rstrip("\n"):
                if ch == " ":
                    continue
                char_freq[ch] = char_freq.get(ch, 0) + 1
                total += 1

    if total == 0:
        return 4000

    sorted_counts = sorted(char_freq.values(), reverse=True)
    target = 0.95 * total
    cumulative = 0
    for i, count in enumerate(sorted_counts):
        cumulative += count
        if cumulative >= target:
            vocab_size = max(1000, (i + 1) * 10)
            print(f"  95% char coverage at {i+1} char types → estimated vocab: {vocab_size}")
            return vocab_size

    return 8000


# ---------------------------------------------------------------------------
# SentencePiece output → @@ convention
# ---------------------------------------------------------------------------

def pieces_to_at_at(pieces: list[str]) -> str:
    """
    Convert SentencePiece pieces to @@ boundary-marker format.

    SentencePiece uses ▁ (U+2581) as a word-boundary prefix on the FIRST
    subword of each word.  We want to output complete words with @@ between
    internal subwords, matching the Moses convention used by segment_morfessor.py.

    Example:
      Input pieces: ['▁map', 'u', '@@', '▁du', 'ng', 'un']   (hypothetical)
      Actually SP output: ['▁mapu', 'dun', 'gun']
      Desired @@-output: 'mapu@@ dun@@ gun'   (within the word)

    Strategy:
      - ▁X starts a new word; the previous word's last piece gets no @@
      - Non-▁ pieces are continuations of the current word → previous gets @@
    """
    if not pieces:
        return ""

    # Each item in result is (text, is_last_in_word)
    # We decide @@-suffix when we see the NEXT piece.
    result_parts = []
    for i, piece in enumerate(pieces):
        if piece.startswith("▁"):
            text = piece[1:]  # strip ▁
            # previous piece (if any) ended its word — already marked in result_parts
            result_parts.append(text)
        else:
            # continuation of current word: previous piece needs @@
            if result_parts:
                result_parts[-1] = result_parts[-1] + "@@"
            result_parts.append(piece)

    return " ".join(result_parts)


def segment_line(sp: spm.SentencePieceProcessor, line: str) -> str:
    """Segment one line and return it in @@ format."""
    line = line.rstrip("\n")
    if not line.strip():
        return "\n"

    pieces = sp.EncodeAsPieces(line)
    return pieces_to_at_at(pieces) + "\n"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sp_bpe(input_files: list[Path], model_prefix: Path,
                 vocab_size: int, character_coverage: float = 1.0):
    """Train a SentencePiece BPE model."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    input_str = ",".join(str(f) for f in input_files)
    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        # Treat whitespace as part of the token (Metaspace-style, matching NLLB)
        add_dummy_prefix=True,
        # Don't split on digits/punctuation by default
        split_digits=False,
    )
    print(f"  Saved SP model → {model_prefix}.model")


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_sp(sp: spm.SentencePieceProcessor, in_file: Path, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(in_file, encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(segment_line(sp, line))
    n = sum(1 for _ in open(out_file))
    print(f"    {in_file.name} → {out_file}  ({n:,} lines)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    bpe_root  = Path(args.bpe_root)

    for approach in args.approaches:
        print(f"\n=== approach: {approach} | mode: {args.mode} ===")

        arn_train = data_root / approach / "cleaned" / "train" / "cleaned" / "src.txt"
        es_train  = data_root / approach / "cleaned" / "train" / "cleaned" / "tgt.txt"

        if not arn_train.exists():
            print(f"  WARNING: {arn_train} not found — skipping")
            continue

        # ------------------------------------------------------------------
        # Train BPE model
        # ------------------------------------------------------------------
        if args.mode == "duan":
            model_prefix = bpe_root / f"{approach}_duan_bpe"
            print(f"Training joint arn+es BPE (vocab_size=5000)...")
            train_sp_bpe(
                input_files=[arn_train, es_train],
                model_prefix=model_prefix,
                vocab_size=5000,
                character_coverage=1.0,
            )
            out_dir_name = "duan_bpe"

        else:  # large
            vocab_size = args.large_vocab
            if vocab_size is None:
                vocab_size = compute_95_coverage_vocab_size(arn_train)
            model_prefix = bpe_root / f"{approach}_large_bpe_{vocab_size}"
            print(f"Training arn-only BPE (vocab_size={vocab_size})...")
            train_sp_bpe(
                input_files=[arn_train],
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=1.0,
            )
            out_dir_name = "large_bpe"

        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_prefix) + ".model")

        # ------------------------------------------------------------------
        # Apply to all splits (arn side only; copy es unchanged)
        # ------------------------------------------------------------------
        for split in SPLITS:
            src_in = data_root / approach / "cleaned" / split / "cleaned" / "src.txt"
            tgt_in = data_root / approach / "cleaned" / split / "cleaned" / "tgt.txt"

            if not src_in.exists():
                print(f"  Skipping {approach}/{split} — src not found")
                continue

            out_dir = data_root / approach / out_dir_name / split
            apply_sp(sp, src_in, out_dir / "src.txt")
            shutil.copy(tgt_in, out_dir / "tgt.txt")
            print(f"    (es copied unchanged)")

        print(f"  Done: {approach}/{out_dir_name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
