#!/usr/bin/env python3
"""
build_human_eval_tsv.py

Build TSV files for human evaluation in mteval (rank eval type).
Samples 50 sentences per direction, stratified by per-sentence chrF++ quartile
on Standard BPE predictions to ensure spread of difficulty.

Outputs:
  human_eval/arn-es.tsv   — 50 arn→es sentences × 2 systems (Standard BPE, Morfessor-VC)
  human_eval/es-arn.tsv   — 50 es→arn sentences × 2 systems (Standard BPE, Morfessor-VC)
  human_eval/rogers_linguistic.tsv  — 25 arn→es sentences for Rogers' qualitative annotation
"""

import random
import csv
from pathlib import Path
from sacrebleu.metrics import CHRF

PREDS_ROOT = Path("/home/it238/nobackup/autodelete/mapudungun/predictions")
DATA_ROOT  = Path("/home/it238/nobackup/autodelete/mapudungun/data-processed")
OUT_DIR    = Path("/home/it238/projects/mapudungun/mapudungun-mt/human_eval")

RANDOM_SEED = 42
N_SAMPLE       = 50
N_LINGUISTIC   = 10  # per direction for Rogers' qualitative annotation (10 arn→es + 10 es→arn)


def load_lines(path: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(path, encoding="utf-8")]


def per_sentence_chrf(preds: list[str], refs: list[str]) -> list[float]:
    """Compute per-sentence chrF++ scores."""
    metric = CHRF(word_order=2)
    scores = []
    for pred, ref in zip(preds, refs):
        score = metric.sentence_score(pred, [ref])
        scores.append(score.score)
    return scores


def stratified_sample(scores: list[float], n: int, seed: int) -> list[int]:
    """Return n indices stratified across chrF++ quartiles."""
    rng = random.Random(seed)
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    n_total = len(indexed)
    n_per_q = n // 4
    # Split into 4 quartiles
    quartiles = [
        indexed[: n_total // 4],
        indexed[n_total // 4 : n_total // 2],
        indexed[n_total // 2 : 3 * n_total // 4],
        indexed[3 * n_total // 4 :],
    ]
    selected = []
    for i, q in enumerate(quartiles):
        # last quartile gets any remainder
        k = n_per_q if i < 3 else n - len(selected)
        selected.extend(rng.sample(q, min(k, len(q))))
    # Return sorted indices for readability
    return sorted(idx for idx, _ in selected)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


def build_direction(
    direction: str,
    src_file: Path,
    ref_file: Path,
    standard_preds_file: Path,
    morfessor_vc_preds_file: Path,
):
    print(f"\n=== {direction} ===")

    src   = load_lines(src_file)
    refs  = load_lines(ref_file)
    std   = load_lines(standard_preds_file)
    mvc   = load_lines(morfessor_vc_preds_file)

    assert len(src) == len(refs) == len(std) == len(mvc), \
        f"Line count mismatch: src={len(src)} refs={len(refs)} std={len(std)} mvc={len(mvc)}"

    # Exclude sentences with runaway/degenerate output in either system
    MAX_OUTPUT_LEN = 800
    valid = [i for i in range(len(src)) if len(std[i]) <= MAX_OUTPUT_LEN and len(mvc[i]) <= MAX_OUTPUT_LEN]
    print(f"  Excluded {len(src) - len(valid)} degenerate sentences (output >{MAX_OUTPUT_LEN} chars); {len(valid)} remain")

    scores_all = per_sentence_chrf(std, refs)
    scores = [scores_all[i] for i in valid]
    print(f"  Standard BPE chrF++ range: {min(scores):.1f}–{max(scores):.1f}, mean: {sum(scores)/len(scores):.1f}")

    valid_indices = stratified_sample(scores, N_SAMPLE, seed=RANDOM_SEED)
    # Map back to original corpus indices
    indices = [valid[i] for i in valid_indices]
    print(f"  Sampled {len(indices)} sentences across chrF++ quartiles")

    # mteval TSV: source + 2 system columns
    rows = []
    for i in indices:
        rows.append({
            "source":        src[i],
            "Standard BPE":  std[i],
            "Morfessor-VC":  mvc[i],
        })

    out_tsv = OUT_DIR / f"{direction}.tsv"
    write_tsv(out_tsv, rows, ["source", "Standard BPE", "Morfessor-VC"])
    return indices, src, refs, std, mvc


def build_rogers_annotation(
    arn_es_indices, src_arn, refs_es, mvc_arn_es,
    es_arn_indices, src_es, refs_arn, mvc_es_arn,
):
    """20-sentence qualitative annotation sheet for Rogers: 10 arn→es + 10 es→arn."""
    rng = random.Random(RANDOM_SEED + 1)

    arn_es_subset = sorted(rng.sample(arn_es_indices, N_LINGUISTIC))
    es_arn_subset = sorted(rng.sample(es_arn_indices, N_LINGUISTIC))

    rows = []
    for i in arn_es_subset:
        rows.append({
            "direction":        "arn→es",
            "source":           src_arn[i],
            "reference":        refs_es[i],
            "MorfessorVC_pred": mvc_arn_es[i],
            "error_type":       "",
            "notes":            "",
        })
    for i in es_arn_subset:
        rows.append({
            "direction":        "es→arn",
            "source":           src_es[i],
            "reference":        refs_arn[i],
            "MorfessorVC_pred": mvc_es_arn[i],
            "error_type":       "",
            "notes":            "",
        })

    fieldnames = ["direction", "source", "reference", "MorfessorVC_pred", "error_type", "notes"]
    out_tsv = OUT_DIR / "rogers_linguistic.tsv"
    write_tsv(out_tsv, rows, fieldnames)
    print(f"\nError type key (for Rogers):")
    print("  morphological_error | code_switching_intrusion | word_order | lexical_gap | fluent_but_wrong | other")


def main():
    data = DATA_ROOT / "blocks" / "cleaned" / "test" / "cleaned"

    # arn→es
    arn_es_indices, src_arn, refs_es, std_arn_es, mvc_arn_es = build_direction(
        direction="arn-es",
        src_file=data / "src.txt",
        ref_file=data / "tgt.txt",
        standard_preds_file=PREDS_ROOT / "3.3B-arn-es_preds.txt",
        morfessor_vc_preds_file=PREDS_ROOT / "nllb-200-3.3B-blocks-arn-es-morfessor_vc_preds.txt",
    )

    # es→arn
    es_arn_indices, src_es, refs_arn, std_es_arn, mvc_es_arn = build_direction(
        direction="es-arn",
        src_file=data / "tgt.txt",
        ref_file=data / "src.txt",
        standard_preds_file=PREDS_ROOT / "3.3B-es-arn_preds.txt",
        morfessor_vc_preds_file=PREDS_ROOT / "3.3B-blocks-es-arn-morfessor_vc_preds.txt",
    )

    # Rogers' qualitative annotation sheet (10 arn→es + 10 es→arn)
    build_rogers_annotation(
        arn_es_indices, src_arn, refs_es, mvc_arn_es,
        es_arn_indices, src_es, refs_arn, mvc_es_arn,
    )

    print("\nDone. Upload arn-es.tsv and es-arn.tsv to mteval as 'rank' eval type.")
    print("Send rogers_linguistic.tsv to Dr. Rogers separately.")


if __name__ == "__main__":
    main()
