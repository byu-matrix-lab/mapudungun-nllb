#!/usr/bin/env python3
"""
segment_morfessor_bpe.py

Morfessor-BPE (Banerjee & Bhattacharyya 2018):
  1. Load Morfessor-segmented arn data (produced by segment_morfessor.py)
  2. Train a SentencePiece BPE model on the morpheme-level corpus —
     each Morfessor morpheme is treated as an independent "word" for BPE training
  3. Apply BPE *within* each morpheme across all splits
  4. Output: Morfessor morpheme boundaries (@@) PLUS within-morpheme BPE boundaries (@@)

This is distinct from Morfessor-VC (segment_morfessor_vc.py), which filters Morfessor
boundaries through NLLB's existing vocabulary without training a new BPE model.

Output: data-processed/{approach}/morfessor_bpe/{split}/{src,tgt}.txt

Usage:
    uv run python scripts/tokenization/segment_morfessor_bpe.py
    uv run python scripts/tokenization/segment_morfessor_bpe.py --vocab-size 4000
    uv run python scripts/tokenization/segment_morfessor_bpe.py --approaches blocks lines
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
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--bpe-root", default=str(BPE_ROOT))
    parser.add_argument("--approaches", nargs="+", default=["blocks", "lines"])
    parser.add_argument("--vocab-size", type=int, default=4000,
                        help="BPE vocabulary size for the morpheme-level model (default: 4000)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parse / reconstruct Morfessor @@ format  (shared logic with segment_cascade.py)
# ---------------------------------------------------------------------------

def parse_morfessor_line(line: str) -> list[list[str]]:
    """
    Parse a Morfessor-segmented line into words, each a list of morpheme strings.

    Input:  "mapu@@ dun@@ gun kimi@@ el"
    Output: [["mapu", "dun", "gun"], ["kimi", "el"]]
    """
    tokens = line.rstrip("\n").split()
    words: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok.endswith("@@"):
            current.append(tok[:-2])
        else:
            current.append(tok)
            words.append(current)
            current = []
    if current:
        words.append(current)
    return words


# ---------------------------------------------------------------------------
# Build morpheme-level training corpus for SentencePiece
# ---------------------------------------------------------------------------

def extract_morphemes(morfessor_file: Path) -> list[str]:
    """Return every morpheme from a Morfessor-segmented file as a flat list."""
    morphemes = []
    with open(morfessor_file, encoding="utf-8") as f:
        for line in f:
            for word in parse_morfessor_line(line):
                morphemes.extend(word)
    return morphemes


def write_morpheme_corpus(morphemes: list[str], tmp_path: Path):
    """
    Write one morpheme per line so SentencePiece treats each morpheme as a
    separate "word".  This prevents BPE from merging across morpheme boundaries.
    """
    with open(tmp_path, "w", encoding="utf-8") as f:
        for m in morphemes:
            if m.strip():
                f.write(m + "\n")


# ---------------------------------------------------------------------------
# Apply BPE within a single morpheme
# ---------------------------------------------------------------------------

def bpe_segment_morpheme(morpheme: str, sp: spm.SentencePieceProcessor) -> list[str]:
    """
    Encode one morpheme with SentencePiece BPE.
    Returns a list of pieces (without the ▁ word-boundary prefix).
    """
    pieces = sp.EncodeAsPieces(morpheme)
    # Strip the leading ▁ that SentencePiece adds to the first piece of a "word"
    result = []
    for i, p in enumerate(pieces):
        result.append(p.lstrip("▁") if i == 0 else p)
    # Filter out empty strings that can arise from stripping ▁ on a lone ▁ token
    return [p for p in result if p] or [morpheme]


def render_word_with_bpe(morphemes: list[str], sp: spm.SentencePieceProcessor) -> str:
    """
    Apply BPE within each Morfessor morpheme and render the whole word in @@ format.

    Example:
      Morfessor morphemes: ["mapudun", "gun"]
      After BPE on "mapudun": ["mapu", "dun"]
      After BPE on "gun":     ["gun"]
      Result: "mapu@@ dun@@ gun"
    """
    all_pieces: list[str] = []
    for i, morpheme in enumerate(morphemes):
        bpe_pieces = bpe_segment_morpheme(morpheme, sp)
        is_last_morpheme = (i == len(morphemes) - 1)
        for j, piece in enumerate(bpe_pieces):
            is_last_piece = is_last_morpheme and (j == len(bpe_pieces) - 1)
            all_pieces.append(piece if is_last_piece else piece + "@@")
    return " ".join(all_pieces)


def segment_line(line: str, sp: spm.SentencePieceProcessor) -> str:
    """Apply Morfessor-BPE segmentation to one line."""
    words = parse_morfessor_line(line)
    if not words:
        return "\n"
    rendered = [render_word_with_bpe(w, sp) for w in words]
    return " ".join(rendered) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    bpe_root  = Path(args.bpe_root)

    for approach in args.approaches:
        print(f"\n=== approach: {approach} | vocab_size: {args.vocab_size} ===")

        morfessor_train = data_root / approach / "morfessor" / "train" / "src.txt"
        if not morfessor_train.exists():
            print(f"  WARNING: {morfessor_train} not found — run segment_morfessor.py first")
            continue

        # ------------------------------------------------------------------
        # Build morpheme-level corpus and train BPE
        # ------------------------------------------------------------------
        print("  Extracting morphemes from Morfessor training output...")
        morphemes = extract_morphemes(morfessor_train)
        print(f"  {len(morphemes):,} morpheme tokens")

        model_prefix = bpe_root / f"{approach}_morfessor_bpe_{args.vocab_size}"
        bpe_root.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            write_morpheme_corpus(morphemes, tmp_path)

        print(f"  Training SentencePiece BPE (vocab_size={args.vocab_size})...")
        spm.SentencePieceTrainer.train(
            input=str(tmp_path),
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            add_dummy_prefix=True,
            split_digits=False,
        )
        tmp_path.unlink(missing_ok=True)
        print(f"  Saved SP model → {model_prefix}.model")

        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_prefix) + ".model")

        # ------------------------------------------------------------------
        # Apply to all splits (arn side only; copy es unchanged)
        # ------------------------------------------------------------------
        for split in SPLITS:
            morfessor_src = data_root / approach / "morfessor" / split / "src.txt"
            cleaned_tgt   = data_root / approach / "cleaned" / split / "cleaned" / "tgt.txt"

            if not morfessor_src.exists():
                print(f"  Skipping {approach}/{split} — morfessor src not found")
                continue

            out_dir = data_root / approach / "morfessor_bpe" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            src_out = out_dir / "src.txt"
            tgt_out = out_dir / "tgt.txt"

            n_lines = 0
            total_boundaries = 0
            with open(morfessor_src, encoding="utf-8") as fin, \
                 open(src_out, "w", encoding="utf-8") as fout:
                for line in fin:
                    out_line = segment_line(line, sp)
                    fout.write(out_line)
                    n_lines += 1
                    total_boundaries += out_line.count("@@")

            shutil.copy(cleaned_tgt, tgt_out)
            print(f"  {split}: {n_lines:,} lines | {total_boundaries:,} @@ boundaries → {src_out}")

        print(f"  Done: {approach}/morfessor_bpe")

    print("\nAll done.")


if __name__ == "__main__":
    main()
