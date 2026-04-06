#!/usr/bin/env python3
"""
segment_cascade.py

NLLB-Constrained Morfessor+BPE Cascade (Kevin Duh, JHU, Mar 2026).

Takes Morfessor-segmented Mapudungun text (with @@ boundary markers) and merges
adjacent morphemes where their concatenation is already a single token in NLLB's
SentencePiece vocabulary.  Boundaries that span two separate NLLB vocabulary items
are kept with @@.

The result has fewer @@ markers than plain Morfessor, reducing double-tokenization:
NLLB's internal BPE no longer sees "mapu@@" as a spurious string — boundaries are
only kept where the split is linguistically meaningful AND corresponds to a genuine
NLLB subword boundary.

Algorithm (greedy left-to-right within each word):
  Given morphemes [m0, m1, m2, ...] in a word:
    candidate = m_i + m_{i+1}
    if "▁" + candidate is a single SentencePiece piece → merge, repeat from m_i
    else → keep boundary, advance to m_{i+1}

Output uses the same @@ convention as segment_morfessor.py, so:
  - desegment() / DESEG_RE in finetune_nllb.py and predict_test.py work unchanged
  - TOKENIZER_APPROACH=morfessor_vc feeds into the same desegmentation pipeline

Input: blocks/morfessor/{split}/src.txt  (Morfessor-segmented arn)
Output: blocks/morfessor_vc/{split}/src.txt   (morfessor_vc-merged arn, fewer @@ markers)
        blocks/morfessor_vc/{split}/tgt.txt   (unchanged es, copied from cleaned)

Usage:
    uv run python scripts/tokenization/segment_cascade.py
    uv run python scripts/tokenization/segment_cascade.py --approaches blocks lines
"""

import argparse
import re
import shutil
from pathlib import Path

import sentencepiece as spm


DATA_ROOT      = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed")
NLLB_SP_MODEL  = (
    Path("/home/it238/.cache/huggingface/hub")
    / "models--facebook--nllb-200-distilled-600M"
    / "snapshots"
    / "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    / "sentencepiece.bpe.model"
)
SPLITS = ["train", "dev", "test"]

# Regex to split a Morfessor-segmented line into tokens (words/morphemes/markers)
# Each "word" in the output is either a morpheme followed by @@ or a final morpheme.
_WORD_RE = re.compile(r"\S+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--sp-model", default=str(NLLB_SP_MODEL),
                        help="Path to NLLB SentencePiece .model file")
    parser.add_argument("--approaches", nargs="+", default=["blocks", "lines"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parse / reconstruct Morfessor @@ format
# ---------------------------------------------------------------------------

def parse_morfessor_line(line: str) -> list[list[str]]:
    """
    Parse a Morfessor-segmented line into words, each a list of morpheme strings.

    Input:  "mapu@@ dun@@ gun kimi@@ el"
    Output: [["mapu", "dun", "gun"], ["kimi", "el"]]
    """
    tokens = line.rstrip("\n").split()
    words: list[list[str]] = []
    current_word: list[str] = []
    for tok in tokens:
        if tok.endswith("@@"):
            current_word.append(tok[:-2])  # strip @@
        else:
            current_word.append(tok)
            words.append(current_word)
            current_word = []
    if current_word:
        words.append(current_word)  # handle truncated line
    return words


def render_word(morphemes: list[str]) -> str:
    """Render a list of morphemes back to @@ format."""
    parts = []
    for i, m in enumerate(morphemes):
        if i < len(morphemes) - 1:
            parts.append(m + "@@")
        else:
            parts.append(m)
    return " ".join(parts)


def render_line(words: list[list[str]]) -> str:
    return " ".join(render_word(w) for w in words) + "\n"


# ---------------------------------------------------------------------------
# NLLB-constrained morfessor_vc merge
# ---------------------------------------------------------------------------

def build_vocab_set(sp: spm.SentencePieceProcessor) -> set[str]:
    """
    Build the set of all SentencePiece pieces (vocabulary items).
    This is faster than calling sp.EncodeAsPieces() per pair.
    """
    vocab = set()
    for i in range(sp.GetPieceSize()):
        vocab.add(sp.IdToPiece(i))
    return vocab


def cascade_merge_word(morphemes: list[str], vocab: set[str]) -> list[str]:
    """
    Greedily merge adjacent morphemes where their concatenation (with ▁ word-start
    prefix) is a single NLLB vocabulary item.  Processes left-to-right; after a
    merge the merged unit is checked against the next morpheme.

    The ▁ prefix represents word-start in NLLB's Metaspace tokenizer.  We check
    "▁" + m1 + m2 for merges at the beginning of a word, and m1 + m2 (no prefix)
    for mid-word merges.  Both must be in vocab for the merge to proceed.
    """
    if len(morphemes) == 1:
        return morphemes

    result = list(morphemes)
    i = 0
    while i < len(result) - 1:
        m1, m2 = result[i], result[i + 1]
        candidate = m1 + m2
        # Check both word-start form (▁candidate) and mid-word form (candidate)
        if ("▁" + candidate in vocab) or (candidate in vocab):
            result[i] = candidate
            del result[i + 1]
            # don't advance i — try merging the new combined token with the next
        else:
            i += 1
    return result


def cascade_line(line: str, vocab: set[str]) -> str:
    """Apply NLLB-constrained morfessor_vc merging to one Morfessor-segmented line."""
    words = parse_morfessor_line(line)
    merged_words = [cascade_merge_word(w, vocab) for w in words]
    return render_line(merged_words)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def boundary_count(line: str) -> int:
    return line.count("@@")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    sp_path   = Path(args.sp_model)

    if not sp_path.exists():
        raise FileNotFoundError(
            f"SentencePiece model not found: {sp_path}\n"
            "Try providing --sp-model with the correct path."
        )

    print(f"Loading NLLB SentencePiece model: {sp_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_path))
    vocab = build_vocab_set(sp)
    print(f"  Vocabulary size: {len(vocab):,} pieces")

    for approach in args.approaches:
        print(f"\n=== approach: {approach} ===")

        morfessor_base = data_root / approach / "morfessor"
        cleaned_base   = data_root / approach / "cleaned"
        cascade_base   = data_root / approach / "morfessor_vc"

        if not morfessor_base.exists():
            print(f"  WARNING: {morfessor_base} not found — run segment_morfessor.py first")
            continue

        for split in SPLITS:
            src_in  = morfessor_base / split / "src.txt"   # Morfessor-segmented arn
            tgt_in  = cleaned_base / split / "cleaned" / "tgt.txt"  # unchanged es

            if not src_in.exists():
                print(f"  Skipping {split} — {src_in} not found")
                continue

            out_dir = cascade_base / split
            out_dir.mkdir(parents=True, exist_ok=True)
            src_out = out_dir / "src.txt"
            tgt_out = out_dir / "tgt.txt"

            total_before = 0
            total_after  = 0
            n_lines = 0

            with open(src_in) as fin, open(src_out, "w") as fout:
                for line in fin:
                    before = boundary_count(line)
                    out_line = cascade_line(line, vocab)
                    after = boundary_count(out_line)
                    total_before += before
                    total_after  += after
                    n_lines += 1
                    fout.write(out_line)

            shutil.copy(tgt_in, tgt_out)

            reduction = (total_before - total_after) / max(total_before, 1) * 100
            print(f"  {split}: {n_lines:,} lines | "
                  f"@@ markers: {total_before:,} → {total_after:,} "
                  f"({reduction:.1f}% reduction)")

    print("\nDone.")


if __name__ == "__main__":
    main()
