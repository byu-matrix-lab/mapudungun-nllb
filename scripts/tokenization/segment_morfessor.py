#!/usr/bin/env python3
"""
segment_morfessor.py

Train a Morfessor model on Mapudungun training data and apply morpheme
segmentation to all splits (train/dev/test) for both blocks and lines approaches.

Morpheme boundaries are marked with @@ (Moses/OpenNMT convention):
    "mapuche" -> "mapu@@ che"  (if Morfessor segments it)
    "inche"   -> "inche"       (if Morfessor keeps it whole)

At inference/eval time, desegment with: text.replace("@@ ", "")

Only the Mapudungun (arn) side is segmented. Spanish (es) is copied unchanged.
Output layout mirrors the input:
    data-processed/{approach}/morfessor/{train,dev,test}/src.txt  (segmented arn)
    data-processed/{approach}/morfessor/{train,dev,test}/tgt.txt  (unchanged es)

Usage:
    uv run python scripts/tokenization/segment_morfessor.py \\
        --data-root ~/nobackup/autodelete/mapudungun/data-processed \\
        --model-out ~/nobackup/autodelete/mapudungun/morfessor_model.bin
"""

import argparse
import re
from pathlib import Path

import morfessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",
                        default="/home/it238/nobackup/autodelete/mapudungun/data-processed")
    parser.add_argument("--model-out",
                        default="/home/it238/nobackup/autodelete/mapudungun/morfessor_model.bin")
    parser.add_argument("--approaches", nargs="+", default=["blocks", "lines"])
    return parser.parse_args()


def tokenize_words(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens (preserve punctuation attached)."""
    return text.split()


def segment_line(line: str, model: morfessor.BaselineModel) -> str:
    """Segment every word in a line using Morfessor, mark boundaries with @@."""
    words = tokenize_words(line.rstrip("\n"))
    segmented_words = []
    for word in words:
        # Morfessor works on lowercase; preserve original case by segmenting
        # the lowercased form and then using boundaries to split the original.
        try:
            constructions, _ = model.viterbi_segment(word.lower())
        except Exception:
            constructions = [word.lower()]

        if len(constructions) == 1:
            segmented_words.append(word)
        else:
            # Re-split the original word at morpheme boundaries
            pieces = []
            pos = 0
            orig = word
            for morpheme in constructions[:-1]:
                end = pos + len(morpheme)
                pieces.append(orig[pos:end] + "@@")
                pos = end
            pieces.append(orig[pos:])
            segmented_words.append(" ".join(pieces))

    return " ".join(segmented_words) + "\n"


def desegment(text: str) -> str:
    """Remove @@ markers to recover original word forms."""
    return re.sub(r"@@ ?", "", text)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model_path = Path(args.model_out)

    # -------------------------------------------------------------------------
    # Collect all Mapudungun training sentences from all approaches for
    # a single unified Morfessor model.
    # -------------------------------------------------------------------------
    print("Collecting Mapudungun training words...")
    io = morfessor.MorfessorIO()
    train_words: dict[str, int] = {}

    for approach in args.approaches:
        train_src = data_root / approach / "cleaned" / "train" / "cleaned" / "src.txt"
        if not train_src.exists():
            print(f"  WARNING: {train_src} not found, skipping")
            continue
        with open(train_src) as f:
            for line in f:
                for word in tokenize_words(line):
                    train_words[word.lower()] = train_words.get(word.lower(), 0) + 1

    print(f"  {len(train_words):,} unique word types from training data")

    # -------------------------------------------------------------------------
    # Train Morfessor
    # -------------------------------------------------------------------------
    print("Training Morfessor...")
    model = morfessor.BaselineModel()
    model.load_data([(count, word) for word, count in train_words.items()], count_modifier=lambda x: x)
    model.train_batch()
    print(f"  Training complete.")

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    io.write_binary_model_file(str(model_path), model)
    print(f"  Saved model to {model_path}")

    # Quick stats: what fraction of word types get split?
    segmented_count = sum(
        1 for w in list(train_words.keys())[:5000]
        if len(model.viterbi_segment(w)[0]) > 1
    )
    print(f"  ~{segmented_count/50:.1f}% of sampled word types get segmented")

    # -------------------------------------------------------------------------
    # Apply segmentation to all splits and approaches
    # -------------------------------------------------------------------------
    splits = ["train", "dev", "test"]

    for approach in args.approaches:
        print(f"\nProcessing approach: {approach}")
        out_base = data_root / approach / "morfessor"

        for split in splits:
            src_in  = data_root / approach / "cleaned" / split / "cleaned" / "src.txt"
            tgt_in  = data_root / approach / "cleaned" / split / "cleaned" / "tgt.txt"

            if not src_in.exists():
                print(f"  Skipping {approach}/{split} — src not found")
                continue

            out_dir = out_base / split
            out_dir.mkdir(parents=True, exist_ok=True)
            src_out = out_dir / "src.txt"
            tgt_out = out_dir / "tgt.txt"

            # Segment Mapudungun source
            with open(src_in) as fin, open(src_out, "w") as fout:
                for line in fin:
                    fout.write(segment_line(line, model))

            # Copy Spanish target unchanged
            import shutil
            shutil.copy(tgt_in, tgt_out)

            n = sum(1 for _ in open(src_out))
            print(f"  {approach}/{split}: {n:,} lines → {src_out}")

    print("\nDone. Sample segmentations:")
    samples = ["inche", "mapuche", "ngütramkafiñ", "kutranken", "mapudungun",
               "ñi", "feymu", "tukulpayayu", "lamngen", "kimün"]
    for w in samples:
        try:
            segs, _ = model.viterbi_segment(w.lower())
        except Exception:
            segs = [w]
        if len(segs) > 1:
            marked = " ".join(s + "@@" if i < len(segs)-1 else s
                              for i, s in enumerate(segs))
        else:
            marked = w
        print(f"  {w:20s} -> {marked}")


if __name__ == "__main__":
    main()
