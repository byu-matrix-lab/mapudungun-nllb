"""Every data-derived number in the v2 paper, computed from raw files.

Sources: full-precision COMET (`v2_comet_full.py`, run first), prediction files (published runs, plus the arn->es Standard BPE retrains
on the original data), the recovered original data, the original segmentation
models (via `v2_segmentation_stats.py`, run first), the Optuna study database,
the human-evaluation sheets, and the 256-token check (`v2c_score.py`).

Merges into results/paper_stats.json["paper_v2"] (keeps other keys). Asserts that
every published value that v2 keeps unchanged reproduces from the files.
"""

import csv
import json
import re
import sqlite3
import statistics
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from scipy import stats as sstats

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/nobackup/archive/usr/it238")
PRED = ARCHIVE / "mapudungun_predictions"
V2_PRED = ARCHIVE / "mapudungun_v2/predictions"
DATA = ARCHIVE / "mapudungun_rebuild/data-processed/blocks"
REGEN = ARCHIVE / "mapudungun_v2/data/blocks/cleaned"
LID_176 = ARCHIVE / "mapudungun_v2/lid/lid.176.bin"
OPTUNA_DB = ARCHIVE / "mapudungun_optuna_bpe.db"
NLLB_SP = (Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/"
           "f8d333a098d19b4fd9a8b18f94170487ad3f821d/sentencepiece.bpe.model")
STATS = REPO / "results" / "paper_stats.json"
N_BOOT, SEED = 10_000, 42

SIZES = ["600M", "1.3B", "3.3B"]
DIRS = ["arn-es", "es-arn"]
NLLB = {"600M": "nllb-600M", "1.3B": "nllb-1.3B", "3.3B": "nllb-200-3.3B"}
FINETUNED = ["Standard BPE", "Joint-5K BPE", "Mono BPE", "Optuna BPE", "Morfessor",
             "Morfessor-VC", "Morfessor-BPE", "UnigramLM"]
STEMS = {
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
LLMS = {"Llama 3.1 8B": "llama-3.1-8b-{d}", "Aya Expanse 8B": "aya-expanse-8b-{d}"}
SEGMENTED = FINETUNED[1:]


def read_lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def key(cond, size, d):
    return f"{cond}|{size}|{d}"


def source_of(cond, size, d):
    """(directory, stem) of the predictions behind a table cell. The arn->es Standard
    BPE cells are the v2 retrains on the original data; everything else is published."""
    if cond == "Standard BPE" and d == "arn-es":
        return V2_PRED, f"v2o-{size}-arn-es"
    if cond in LLMS:
        return PRED, LLMS[cond].format(d=d)
    return PRED, STEMS[cond].format(size=size, d=d, nllb=NLLB[size])


def comet_of(stem, comet_full):
    """Full-precision COMET with the raw source (v2_comet_full.py)."""
    return comet_full[stem]


def bootstrap(xa, xb):
    """Paired bootstrap of the mean sentence-level difference b - a (the paper's metric)."""
    rng = np.random.default_rng(SEED)
    d = xb - xa
    diffs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(p),
            "p_below_resolution": bool(p < 1 / N_BOOT)}


def published_table(name):
    """Rows of a published LaTeX table (main.tex), for asserting unchanged cells."""
    tex = (REPO / "latex/main.tex").read_text(encoding="utf-8")
    body = tex[tex.index(f"\\label{{tab:{name}}}") - 3000: tex.index(f"\\label{{tab:{name}}}")]
    out = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) >= 7 and cells[0] in FINETUNED + ["Zero-shot NLLB"]:
            out[cells[0]] = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c) for c in cells[1:]]
    return out


def main():
    chrf = CHRF(word_order=2)
    test = {"arn": read_lines(DATA / "cleaned/test/cleaned/src.txt"),
            "es": read_lines(DATA / "cleaned/test/cleaned/tgt.txt")}
    refs = {"arn-es": test["es"], "es-arn": test["arn"]}
    comet_full = json.loads((REPO / "results/v2_comet_full.json").read_text(encoding="utf-8"))
    old_cells = {(c["condition"], c["size"], c["direction"]): c["table"]
                 for c in json.loads((REPO / "results/v2_phaseA_audit.json").read_text())["cells"]}

    # ---- every table cell -------------------------------------------------
    cells, sent = {}, {}
    systems = [(c, s, d) for c in FINETUNED + ["Zero-shot NLLB"] for s in SIZES for d in DIRS]
    systems += [(c, "8B", d) for c in LLMS for d in DIRS]
    for cond, size, d in systems:
        base, stem = source_of(cond, size, d)
        rows = json.loads((base / f"{stem}_results.json").read_text(encoding="utf-8"))
        preds = [r["pred"] for r in rows]
        assert [r["ref"] for r in rows] == refs[d], f"{stem}: references differ from the test set"
        st = np.array(chrf._extract_corpus_statistics(preds, [refs[d]]))
        sc = np.array([chrf._compute_score_from_stats(list(x)).score for x in st])
        k = key(cond, size, d)
        sent[k] = sc
        cells[k] = {
            "condition": cond, "size": size, "direction": d, "file": str(base / stem),
            "chrfpp_mean": float(sc.mean()),
            "chrfpp_corpus": chrf._compute_score_from_stats(list(st.sum(axis=0))).score,
            "bleu": sacrebleu.corpus_bleu(preds, [refs[d]]).score,
            "comet": comet_of(stem, comet_full),
            "changed_in_v2": cond == "Standard BPE" and d == "arn-es",
        }
        if not cells[k]["changed_in_v2"]:
            assert abs(round(cells[k]["chrfpp_mean"], 2) - old_cells[(cond, size, d)]) < 0.006, k

    # Unchanged BLEU and COMET cells must reproduce the published appendix tables.
    mismatches = []
    for name, field, ndp in (("bleu-results", "bleu", 2), ("comet-results", "comet", 3)):
        pub = published_table(name)
        for cond, vals in pub.items():
            for (s, d), v in zip([(s, d) for d in DIRS for s in SIZES], vals):
                c = cells[key(cond, s, d)]
                changed = c["changed_in_v2"] or (field == "comet" and d == "arn-es" and cond in SEGMENTED)
                if not changed and f"{c[field]:.{ndp}f}" != v:
                    mismatches.append(f"{name} {cond} {s} {d}: computed {c[field]:.{ndp}f}, published {v}")
    # v1 rounded COMET from 4-decimal summaries, so these four ties at the 4th decimal were
    # printed 0.001 off (three truncated, one rounded twice). Verified 2026-11-11: the
    # full-precision rescore matches all 37 unchanged 4-decimal summaries within 0.00005.
    rounding_fixed = {"comet-results Standard BPE 3.3B es-arn", "comet-results Optuna BPE 1.3B es-arn",
                      "comet-results Morfessor 3.3B es-arn", "comet-results Zero-shot NLLB 1.3B es-arn"}
    unexpected = [m for m in mismatches if m.split(":")[0] not in rounding_fixed]
    assert not unexpected, "unchanged cells do not reproduce:\n" + "\n".join(unexpected)
    assert len(mismatches) == len(rounding_fixed), "expected rounding corrections changed:\n" + "\n".join(mismatches)

    # ---- best-in-column (fine-tuned only) ---------------------------------
    best = {}
    for field in ("chrfpp_mean", "bleu", "comet"):
        for s in SIZES:
            for d in DIRS:
                col = {c: round(cells[key(c, s, d)][field], 3 if field == "comet" else 2) for c in FINETUNED}
                top = max(col.values())
                best[f"{field}|{s}|{d}"] = [c for c, v in col.items() if v == top]

    # ---- comparisons (paired bootstrap on the mean, uncorrected) ----------
    pairs = {
        "vc_minus_std_600M": (("Standard BPE", "600M"), ("Morfessor-VC", "600M")),
        "vc_minus_std_1.3B": (("Standard BPE", "1.3B"), ("Morfessor-VC", "1.3B")),
        "vc_minus_std_3.3B": (("Standard BPE", "3.3B"), ("Morfessor-VC", "3.3B")),
        "vc600_minus_std3.3B": (("Standard BPE", "3.3B"), ("Morfessor-VC", "600M")),
        "vc_minus_morf_600M": (("Morfessor", "600M"), ("Morfessor-VC", "600M")),
        "vc_minus_morf_1.3B": (("Morfessor", "1.3B"), ("Morfessor-VC", "1.3B")),
        "vc_minus_morf_3.3B": (("Morfessor", "3.3B"), ("Morfessor-VC", "3.3B")),
    }
    comparisons = {name: bootstrap(sent[key(*a, "arn-es")], sent[key(*b, "arn-es")])
                   for name, (a, b) in pairs.items()}

    # ---- margins over the best LLM and condition ranges -------------------
    def m(c, s, d):
        return cells[key(c, s, d)]["chrfpp_mean"]
    llm = {d: max(m(c, "8B", d) for c in LLMS) for d in DIRS}
    margins = {
        "best_llm": llm,
        "min_finetuned_over_llm": min(m(c, s, d) - llm[d] for c in FINETUNED for s in SIZES for d in DIRS),
        "min_finetuned_600M_arn_es_over_llm": min(m(c, "600M", "arn-es") - llm["arn-es"] for c in FINETUNED),
        "best_finetuned_arn_es_over_llm": max(m(c, s, "arn-es") for c in FINETUNED for s in SIZES) - llm["arn-es"],
        "std600_arn_es_over_llm": m("Standard BPE", "600M", "arn-es") - llm["arn-es"],
    }
    others_600 = [m(c, "600M", "arn-es") for c in SEGMENTED]
    ranges = {
        "others_600M_arn_es_min": min(others_600), "others_600M_arn_es_max": max(others_600),
        "std_gap_to_best_arn_es_max": max(max(m(c, s, "arn-es") for c in FINETUNED) - m("Standard BPE", s, "arn-es")
                                          for s in SIZES),
    }

    # ---- human evaluation (rank 1 = preferred) ----------------------------
    human = {}
    for d in DIRS:
        ranks = list(csv.reader(open(REPO / f"human_eval/{d} results.tsv", encoding="utf-8"), delimiter="\t"))[1:]
        outs = list(csv.reader(open(REPO / f"human_eval/{d}.tsv", encoding="utf-8"), delimiter="\t"))[1:]
        assert [r[0] for r in ranks] == [o[0] for o in outs]
        ties = [i for i, r in enumerate(ranks) if r[1] == r[2]]
        human[d] = {"n": len(ranks), "std_preferred": sum(r[1] < r[2] for r in ranks),
                    "vc_preferred": sum(r[2] < r[1] for r in ranks), "ties": len(ties),
                    "ties_identical_outputs": sum(outs[i][1] == outs[i][2] for i in ties)}

    # ---- Optuna ------------------------------------------------------------
    db = sqlite3.connect(f"file:{OPTUNA_DB}?mode=ro", uri=True)
    trials = db.execute("select state from trials").fetchall()
    best_vocab = db.execute(
        "select p.param_value from trials t join trial_values v on v.trial_id=t.trial_id "
        "join trial_params p on p.trial_id=t.trial_id where t.state='COMPLETE' "
        "order by v.value desc, t.number asc limit 1").fetchone()[0]
    slurm = (REPO / "configs/slurm/optuna_bpe_trial.slurm").read_text()
    seg = json.loads(STATS.read_text(encoding="utf-8"))["paper_v2"]["segmentation"]
    assert seg["conditions"]["Optuna BPE"]["test_input_reproduced"] == 9382  # 5,829-piece model
    optuna = {"n_trials": len(trials), "n_complete": sum(s == "COMPLETE" for (s,) in trials),
              "epochs_per_trial": int(re.search(r"--epochs\s+(\d+)", slurm).group(1)),
              "reported_best_vocab": int(best_vocab), "vocab_used": 5829}

    # ---- data statistics (recovered original data) -------------------------
    data = {}
    for split in ("train", "dev", "test"):
        for side, f in (("arn", "src"), ("es", "tgt")):
            lines = read_lines(DATA / f"cleaned/{split}/cleaned/{f}.txt")
            toks = [w for l in lines for w in l.split()]
            data[f"{split}|{side}"] = {"sents": len(lines), "tokens": len(toks), "types": len(set(toks)),
                                       "avg_len": len(toks) / len(lines)}
    regen_train = len(read_lines(REGEN / "train/cleaned/src.txt"))

    # ---- code-switching ------------------------------------------------------
    extracted = [l for s in ("train", "dev", "test") for l in read_lines(DATA / f"{s}.arn")]
    spa = sum(bool(re.search(r"\*?SPA>", l)) for l in extracted)
    import fasttext
    lid = fasttext.load_model(str(LID_176))

    def es_frac(sentence):
        words = re.findall(r"\b\w+\b", sentence, flags=re.UNICODE)
        return sum(lid.predict(w.lower(), k=1)[0][0] == "__label__es" for w in words) / len(words) if words else 0.0
    vc33 = sent[key("Morfessor-VC", "3.3B", "arn-es")]
    frac = np.array([es_frac(s) for s in test["arn"]])
    pr, pp = sstats.pearsonr(frac, vc33)
    sr, _ = sstats.spearmanr(frac, vc33)
    buckets = {"none": frac == 0, "light": (frac > 0) & (frac <= 0.2), "heavy": frac > 0.2}
    codeswitch = {"elan_spa_share_pct": 100 * spa / len(extracted), "n_extracted": len(extracted),
                  "lid_model": "fastText lid.176", "pearson": pr, "pearson_p": pp, "pearson_r2": pr ** 2,
                  "spearman": sr, **{f"{b}_n": int(mask.sum()) for b, mask in buckets.items()},
                  **{f"{b}_mean": float(vc33[mask].mean()) for b, mask in buckets.items()}}

    # ---- 256-token check (regenerated training data, one seed) ---------------
    l256 = json.loads(STATS.read_text(encoding="utf-8"))["v2"]["len256"]["comparisons"]
    corpus_diffs = [v["corpus_chrfpp"] for v in l256.values()]
    len256 = {"max_abs_corpus_diff": max(abs(c["diff"]) for c in corpus_diffs),
              "all_corpus_ci_include_zero": all(c["ci95"][0] <= 0 <= c["ci95"][1] for c in corpus_diffs),
              "regen_train_pairs": regen_train, "orig_train_pairs": data["train|arn"]["sents"],
              "regen_larger_pct": 100 * (regen_train - data["train|arn"]["sents"]) / data["train|arn"]["sents"]}
    assert len256["all_corpus_ci_include_zero"]

    # ---- worked example (Table 2): kutrankautulay ----------------------------
    import morfessor
    import sentencepiece as spm
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "tokenization"))
    import segment_morfessor as sm
    import segment_morfessor_vc as vcmod
    sp = spm.SentencePieceProcessor()
    sp.Load(str(NLLB_SP))
    morf_model = morfessor.MorfessorIO().read_binary_model_file(str(ARCHIVE / "mapudungun_morfessor_model.bin"))
    word = "kutrankautulay"
    morf = sm.segment_line(word, morf_model).strip()
    vcs = vcmod.cascade_line(morf, vcmod.build_vocab_set(sp)).strip()
    example = {"word": word, "standard": " ".join(sp.EncodeAsPieces(word)), "morfessor": morf, "morfessor_vc": vcs,
               "nllb_input_morfessor": " ".join(sp.EncodeAsPieces(morf)),
               "nllb_input_morfessor_vc": " ".join(sp.EncodeAsPieces(vcs)),
               "kau_word_initial_id": sp.PieceToId("\u2581kau"), "kau_mid_word_id": sp.PieceToId("kau")}

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    stats["paper_v2"] = {**stats.get("paper_v2", {}), "cells": cells, "best_in_column": best,
                         "comparisons": comparisons, "margins": margins, "ranges": ranges,
                         "human_eval": human, "optuna": optuna, "data": data, "codeswitch": codeswitch,
                         "len256": len256, "example": example, "comet_rounding_corrected": mismatches, "bootstrap": {"n": N_BOOT, "seed": SEED}}
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: stats["paper_v2"][k] for k in
                      ("comparisons", "margins", "ranges", "human_eval", "optuna", "codeswitch", "len256", "example")},
                     indent=1, default=float))
    print(f"merged into {STATS}")


if __name__ == "__main__":
    main()
