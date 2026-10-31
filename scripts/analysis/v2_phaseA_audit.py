"""Phase A audit for the v2 correction (see notes/v2_brief.md).

1. Reconstruct the original test set from the surviving *_results.json files
   and check that every file agrees on src/ref (after stripping the @@
   segmentation markers that pre-segmented conditions store in src/ref).
2. Check that *_preds.txt files agree with the pred field in *_results.json.
3. Leakage check: overlap of the original test set with the regenerated splits
   in morphotok/experiments/data_pipeline/mapudungun/cleaned/.
4. Reproduce every cell of results/metrics_table.tsv from the surviving
   predictions via an explicit file -> cell mapping, under both mean
   per-sentence chrF++ and corpus chrF++.

Writes results/v2_phaseA_audit.json (does not touch any paper_stats.json).
"""

import json
import re
from collections import Counter
from pathlib import Path

from sacrebleu.metrics import CHRF

PRED_DIR = Path("/nobackup/archive/usr/it238/mapudungun_predictions")
REGEN_DIR = Path("/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/cleaned")
REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "results" / "metrics_table.tsv"
OUT = REPO / "results" / "v2_phaseA_audit.json"

DESEG_RE = re.compile(r"@@ ?")
SIZES = ["600M", "1.3B", "3.3B"]
DIRS = {"arn-es": "arn_es_chrf", "es-arn": "es_arn_chrf"}
NLLB_PREFIX = {"600M": "nllb-600M", "1.3B": "nllb-1.3B", "3.3B": "nllb-200-3.3B"}

# Which prediction file backs which table cell. Files not listed here (e.g.
# 600M-blocks-arn-es-morfessor_bpe, 3.3B-blocks-*-unigram_lm, arn-es, es-arn)
# are earlier runs that the table does not use.
CELL_FILES = {
    "Standard BPE": "{size}-{d}",
    "Joint-5K BPE": "{size}-blocks-{d}-joint_bpe",
    "Mono BPE": "{size}-blocks-{d}-mono_bpe",
    "Optuna BPE": "{nllb}-blocks-{d}-optuna_bpe",
    "Morfessor": "{size}-{d}-morfessor",
    "Morfessor-VC": "{size}-blocks-{d}-morfessor_vc",
    "Morfessor-BPE": "{nllb}-blocks-{d}-morfessor_bpe",
    "UnigramLM": "{nllb}-blocks-{d}-unigram_lm",
    "Zero-shot NLLB": "zero-shot-nllb-{size}-{d}",
}
LLM_FILES = {"Llama 3.1 8B": "llama-3.1-8b-{d}", "Aya Expanse 8B": "aya-expanse-8b-{d}"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def deseg(s):
    return " ".join(DESEG_RE.sub("", s).split())


def load(stem):
    return json.loads((PRED_DIR / f"{stem}_results.json").read_text(encoding="utf-8"))


def main():
    report = {}
    chrf = CHRF(word_order=2)

    # ---- 1. Test-set agreement ----
    files = sorted(PRED_DIR.glob("*_results.json"))
    report["n_results_files"] = len(files)
    base = None
    disagree = []
    n_rows = Counter()
    for p in files:
        rows = json.loads(p.read_text(encoding="utf-8"))
        n_rows[len(rows)] += 1
        if "arn-es" in p.name:
            pairs = [(deseg(r["src"]), deseg(r["ref"])) for r in rows]
        else:
            pairs = [(deseg(r["ref"]), deseg(r["src"])) for r in rows]
        if base is None:
            base, base_name = pairs, p.name
        elif pairs != base:
            disagree.append(p.name)
    report["row_counts"] = dict(n_rows)
    report["test_set_reference_file"] = base_name
    report["files_disagreeing_after_deseg"] = disagree
    report["original_test_n"] = len(base)
    report["original_test_unique_pairs"] = len(set(base))
    report["original_test_unique_arn"] = len({a for a, _ in base})

    # ---- 2. preds.txt vs results.json ----
    mism = []
    for p in files:
        txt = p.with_name(p.name.replace("_results.json", "_preds.txt"))
        rows = json.loads(p.read_text(encoding="utf-8"))
        lines = read_lines(txt)
        if lines != [r["pred"] for r in rows]:
            mism.append(p.name)
    report["preds_txt_mismatches"] = mism

    # ---- 3. Leakage vs regenerated splits ----
    regen = {s: list(zip(read_lines(REGEN_DIR / f"{s}.arn"), read_lines(REGEN_DIR / f"{s}.es")))
             for s in ["train", "dev", "test"]}
    report["regen_sizes"] = {s: len(v) for s, v in regen.items()}
    orig = set(base)
    train = set(regen["train"])
    in_train = sorted(orig & train)
    report["leakage"] = {
        "orig_test_pairs_in_regen_train": len(in_train),
        "orig_test_pairs_in_regen_dev": len(orig & set(regen["dev"])),
        "orig_test_pairs_in_regen_test": len(orig & set(regen["test"])),
        "orig_test_arn_in_regen_train": len({a for a, _ in orig} & {a for a, _ in train}),
        "orig_test_es_in_regen_train": len({e for _, e in orig} & {e for _, e in train}),
        "overlap_pair_arn_word_lengths": dict(Counter(len(a.split()) for a, _ in in_train)),
        "baseline_regen_test_pairs_in_regen_train": len(set(regen["test"]) & train),
        "overlap_pairs": in_train,
    }

    # ---- 4. Reproduce table cells ----
    table = {}
    for line in read_lines(TABLE)[1:]:
        cond, size, a, e = line.split("\t")
        table[(cond, size)] = {"arn-es": float(a), "es-arn": float(e)}

    cells = []
    for (cond, size), vals in table.items():
        for d in DIRS:
            if cond in LLM_FILES:
                stem = LLM_FILES[cond].format(d=d)
            else:
                stem = CELL_FILES[cond].format(size=size, d=d, nllb=NLLB_PREFIX[size])
            rows = load(stem)
            preds = [r["pred"] for r in rows]
            refs = [deseg(r["ref"]) for r in rows]
            mean_sent = round(sum(r["chrf"] for r in rows) / len(rows), 2)
            corpus = round(chrf.corpus_score(preds, [refs]).score, 2)
            cells.append({
                "condition": cond, "size": size, "direction": d, "file": stem,
                "table": vals[d], "mean_sentence_chrfpp": mean_sent, "corpus_chrfpp": corpus,
                "mean_matches_table": abs(mean_sent - vals[d]) < 0.006,
                "corpus_matches_table": abs(corpus - vals[d]) < 0.006,
            })
    report["cells"] = cells
    report["n_cells"] = len(cells)
    report["n_cells_mean_match"] = sum(c["mean_matches_table"] for c in cells)
    report["n_cells_corpus_match"] = sum(c["corpus_matches_table"] for c in cells)

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    for k in ["n_results_files", "row_counts", "files_disagreeing_after_deseg", "original_test_n",
              "original_test_unique_pairs", "preds_txt_mismatches", "regen_sizes",
              "n_cells", "n_cells_mean_match", "n_cells_corpus_match"]:
        print(k, report[k])
    print({k: v for k, v in report["leakage"].items() if k != "overlap_pairs"})
    for c in cells:
        if not c["mean_matches_table"]:
            print("MEAN MISMATCH", c)

    assert not disagree, "test sets disagree"
    assert not mism, "preds.txt/results.json mismatch"
    assert report["n_cells_mean_match"] == len(cells), "table cell not reproduced"


if __name__ == "__main__":
    main()
