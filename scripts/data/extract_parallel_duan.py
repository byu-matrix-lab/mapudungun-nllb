"""
extract_parallel_duan.py

Extract parallel Mapudungun–Spanish pairs from the Duan et al. (2020) AVENUE corpus
(translation-clean/ directory from github.com/mingjund/mapudungun-corpus).

Two extraction approaches:
  --approach lines   Pair each M: line with its adjacent C: line positionally within
                     each identifier block (~266K pairs, directly comparable to Duan et al.)
  --approach blocks  Concatenate all M: lines and all C: lines within each identifier
                     block into a single pair (~94K pairs, complete sentences)

Output files per split (train/dev/test):
  <output>/<approach>/train.arn, train.es
  <output>/<approach>/dev.arn,   dev.es
  <output>/<approach>/test.arn,  test.es

Usage:
  python scripts/data/extract_parallel_duan.py \\
      --corpus /home/it238/projects/mapudungun/mapudungun-corpus \\
      --output /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --approach lines

  python scripts/data/extract_parallel_duan.py \\
      --corpus /home/it238/projects/mapudungun/mapudungun-corpus \\
      --output /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --approach blocks
"""

import argparse
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract parallel Mapudungun–Spanish pairs from the Duan et al. corpus."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to the mapudungun-corpus repo root (contains translation-clean/ and dataset_splits/).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root output directory. Results written to <output>/<approach>/.",
    )
    parser.add_argument(
        "--approach",
        choices=["lines", "blocks"],
        required=True,
        help=(
            "'lines': pair M:/C: lines positionally within each block (~266K pairs). "
            "'blocks': concatenate all M:/C: lines per block into one pair (~94K pairs)."
        ),
    )
    return parser.parse_args()


def parse_file(filepath: Path) -> list[tuple[str, str]]:
    """
    Parse one translation-clean .txt file and return a list of (mapudungun, spanish) pairs.

    Each identifier block starts with a line ending in ':' (the utterance ID), followed
    by interleaved M: / C: lines. Multiple M:/C: pairs can appear per block.

    Returns pairs according to the global approach set at module level.
    """
    pairs = []
    current_m_lines = []
    current_c_lines = []
    in_block = False

    def flush_block():
        """Emit pair(s) from the current accumulated block."""
        if not current_m_lines and not current_c_lines:
            return
        if APPROACH == "lines":
            # Positional pairing: zip M: lines with C: lines
            for m, c in zip(current_m_lines, current_c_lines):
                if m and c:
                    pairs.append((m, c))
        else:  # blocks
            m_text = " ".join(current_m_lines)
            c_text = " ".join(current_c_lines)
            if m_text and c_text:
                pairs.append((m_text, c_text))

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                # Skip header/comment lines
                if line.startswith(";") or line.startswith("//"):
                    continue

                # Identifier line: ends with ':' and contains no leading whitespace
                # Pattern: word chars, hyphens, underscores, digits, then ':'
                if re.match(r"^[\w\-]+:", line) and not line.startswith("M:") and not line.startswith("C:"):
                    # New block — flush the previous one
                    flush_block()
                    current_m_lines = []
                    current_c_lines = []
                    in_block = True
                    continue

                if not in_block:
                    continue

                if line.startswith("M:"):
                    text = line[2:].strip()
                    if text:
                        current_m_lines.append(text)
                elif line.startswith("C:"):
                    text = line[2:].strip()
                    if text:
                        current_c_lines.append(text)

        # Flush the final block
        flush_block()
    except Exception as e:
        print(f"  Warning: could not read {filepath}: {e}")

    return pairs


def load_split_list(split_file: Path) -> list[str]:
    """Return list of file basenames from a dataset_splits/mt/ list file."""
    basenames = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                basenames.append(name)
    return basenames


def extract_split(
    basenames: list[str],
    translation_clean_dir: Path,
    split_name: str,
) -> tuple[list[str], list[str]]:
    """Extract all pairs for one split. Returns (arn_lines, es_lines)."""
    arn_lines = []
    es_lines = []
    missing = 0

    for name in basenames:
        filepath = translation_clean_dir / f"{name}.txt"
        if not filepath.exists():
            print(f"  Missing: {filepath.name}")
            missing += 1
            continue
        pairs = parse_file(filepath)
        for m, c in pairs:
            arn_lines.append(m)
            es_lines.append(c)

    print(
        f"  {split_name}: {len(arn_lines):,} pairs from "
        f"{len(basenames) - missing}/{len(basenames)} files"
        + (f" ({missing} missing)" if missing else "")
    )
    return arn_lines, es_lines


def write_split(out_dir: Path, split_name: str, arn_lines: list[str], es_lines: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    arn_path = out_dir / f"{split_name}.arn"
    es_path = out_dir / f"{split_name}.es"
    with open(arn_path, "w", encoding="utf-8") as f:
        f.write("\n".join(arn_lines) + "\n")
    with open(es_path, "w", encoding="utf-8") as f:
        f.write("\n".join(es_lines) + "\n")
    print(f"  Wrote {arn_path.name} and {es_path.name}")


def main():
    args = parse_args()

    global APPROACH
    APPROACH = args.approach

    corpus_dir = Path(args.corpus)
    translation_clean_dir = corpus_dir / "translation-clean"
    splits_dir = corpus_dir / "dataset_splits" / "mt"
    out_dir = Path(args.output) / args.approach

    # Verify corpus structure
    for d in [translation_clean_dir, splits_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    print(f"Extraction approach : {APPROACH}")
    print(f"Corpus              : {corpus_dir}")
    print(f"Output directory    : {out_dir}")
    print()

    splits = {
        "train": splits_dir / "training_files.txt",
        "dev":   splits_dir / "dev_files.txt",
        "test":  splits_dir / "test_files.txt",
    }

    total_arn = total_es = 0
    for split_name, split_file in splits.items():
        if not split_file.exists():
            print(f"  Skipping {split_name}: {split_file} not found")
            continue
        print(f"Processing {split_name} split ({split_file.name})...")
        basenames = load_split_list(split_file)
        arn_lines, es_lines = extract_split(basenames, translation_clean_dir, split_name)
        write_split(out_dir, split_name, arn_lines, es_lines)
        total_arn += len(arn_lines)
        total_es += len(es_lines)
        print()

    print(f"Done. Total pairs written: {total_arn:,}")
    assert total_arn == total_es, "Mismatch between arn and es line counts!"


if __name__ == "__main__":
    main()
