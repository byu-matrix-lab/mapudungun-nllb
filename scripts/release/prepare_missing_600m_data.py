#!/usr/bin/env python3
"""Recreate the two segmented datasets needed for missing 600M releases.

This applies the preserved paper-era segmenters to the recovered, exact cleaned
corpus.  It never trains or changes a segmenter.  The output manifest records
the source and output hashes used by the reconstruction runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import morfessor
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.tokenization.segment_bpe import segment_line as segment_sp_line
from scripts.tokenization.segment_morfessor import segment_line as segment_morfessor_line
from scripts.tokenization.segment_morfessor_bpe import segment_line as segment_morfessor_bpe_line


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Recovered data-processed root containing blocks/cleaned.")
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="Directory containing the preserved paper segmenters.")
    parser.add_argument("--morfessor-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sp(path: Path) -> spm.SentencePieceProcessor:
    if not path.is_file():
        raise FileNotFoundError(path)
    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(path)):
        raise RuntimeError(f"could not load SentencePiece model: {path}")
    return processor


def write_segmented(source: Path, destination: Path, transform) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(transform(line))
            count += 1
    return count


def main() -> None:
    args = parse_args()
    clean_root = args.data_root / "blocks" / "cleaned"
    morfessor_path = args.morfessor_model.resolve()
    mbpe_path = (args.asset_root / "blocks_morfessor_bpe_4000.model").resolve()
    unigram_path = (args.asset_root / "blocks_unigram_lm_1000.model").resolve()

    for path in (morfessor_path, mbpe_path, unigram_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    morph = morfessor.MorfessorIO().read_binary_model_file(str(morfessor_path))
    mbpe = load_sp(mbpe_path)
    unigram = load_sp(unigram_path)
    manifest: dict[str, object] = {
        "purpose": "reconstruct missing paper-era 600M release checkpoints",
        "assets": {str(p): sha256(p) for p in (morfessor_path, mbpe_path, unigram_path)},
        "files": {},
    }

    for split in SPLITS:
        arn = clean_root / split / "cleaned" / "src.txt"
        es = clean_root / split / "cleaned" / "tgt.txt"
        if not arn.is_file() or not es.is_file():
            raise FileNotFoundError(f"missing cleaned {split} split under {clean_root}")
        arn_count = sum(1 for _ in arn.open(encoding="utf-8"))
        es_count = sum(1 for _ in es.open(encoding="utf-8"))
        if arn_count != es_count:
            raise ValueError(f"{split}: source/target mismatch: {arn_count} != {es_count}")

        morph_lines = [segment_morfessor_line(line, morph) for line in arn.open(encoding="utf-8")]

        mbpe_dir = args.data_root / "blocks" / "morfessor_bpe" / split
        mbpe_dir.mkdir(parents=True, exist_ok=True)
        with (mbpe_dir / "src.txt").open("w", encoding="utf-8") as dst:
            for line in morph_lines:
                dst.write(segment_morfessor_bpe_line(line, mbpe))
        shutil.copyfile(es, mbpe_dir / "tgt.txt")

        unigram_dir = args.data_root / "blocks" / "unigram_lm" / split
        write_segmented(arn, unigram_dir / "src.txt", lambda line: segment_sp_line(unigram, line))
        shutil.copyfile(es, unigram_dir / "tgt.txt")

        for condition_dir in (mbpe_dir, unigram_dir):
            out_src = condition_dir / "src.txt"
            out_tgt = condition_dir / "tgt.txt"
            out_count = sum(1 for _ in out_src.open(encoding="utf-8"))
            if out_count != arn_count:
                raise ValueError(f"{out_src}: expected {arn_count} lines, got {out_count}")
            manifest["files"][str(out_src)] = {"lines": out_count, "sha256": sha256(out_src)}
            manifest["files"][str(out_tgt)] = {"lines": out_count, "sha256": sha256(out_tgt)}

        manifest["files"][str(arn)] = {"lines": arn_count, "sha256": sha256(arn)}
        manifest["files"][str(es)] = {"lines": es_count, "sha256": sha256(es)}

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
