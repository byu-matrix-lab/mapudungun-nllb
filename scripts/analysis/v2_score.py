"""v2 scoring: corpus chrF++ (primary), mean sentence chrF++ (the published
version's metric), corpus BLEU, and paired bootstrap significance on both chrF++
statistics. See notes/v2_brief.md.

Systems: the three retrained arn->es Standard BPE models (mapudungun_v2/predictions)
plus every surviving original prediction file used by the published table.
All systems are scored against the ORIGINAL 9,382-pair test set
(mapudungun_v2/data/blocks/cleaned/test/cleaned), raw references.

Merges its results into results/paper_stats.json under "v2" (never overwrites
other keys).
"""

import json
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/nobackup/archive/usr/it238/mapudungun_predictions")
V2_PRED = Path("/nobackup/archive/usr/it238/mapudungun_v2/predictions")
TEST = Path("/nobackup/archive/usr/it238/mapudungun_v2/data/blocks/cleaned/test/cleaned")
STATS = REPO / "results" / "paper_stats.json"
N_BOOT = 10_000
SEED = 42

SIZES = ["600M", "1.3B", "3.3B"]
NLLB = {"600M": "nllb-600M", "1.3B": "nllb-1.3B", "3.3B": "nllb-200-3.3B"}
# condition -> prediction-file stem (same mapping verified in v2_phaseA_audit.py)
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
# Rows replaced in v2: the original arn->es Standard checkpoints were trained on
# untagged data (transformers 5.2.0) and are superseded by the retrains.
V2_STEMS = {("Standard BPE", s, "arn-es"): f"v2-{s}-arn-es" for s in SIZES}

# Comparisons reported (a, b): diff = b - a. Covers every p-value in the
# published paper plus Morfessor-VC vs Standard at every size.
COMPARISONS = [
    (("Standard BPE", "3.3B", "arn-es"), ("Morfessor-VC", "600M", "arn-es")),  # abstract / 5x claim
    (("Standard BPE", "600M", "arn-es"), ("Morfessor-VC", "600M", "arn-es")),
    (("Standard BPE", "1.3B", "arn-es"), ("Morfessor-VC", "1.3B", "arn-es")),
    (("Standard BPE", "3.3B", "arn-es"), ("Morfessor-VC", "3.3B", "arn-es")),
    (("Standard BPE", "3.3B", "arn-es"), ("Morfessor", "600M", "arn-es")),
    (("Standard BPE", "3.3B", "arn-es"), ("Morfessor", "3.3B", "arn-es")),
    (("Morfessor", "600M", "arn-es"), ("Morfessor-VC", "600M", "arn-es")),
    (("Morfessor", "1.3B", "arn-es"), ("Morfessor-VC", "1.3B", "arn-es")),
    (("Morfessor", "3.3B", "arn-es"), ("Morfessor-VC", "3.3B", "arn-es")),
    (("Joint-5K BPE", "3.3B", "arn-es"), ("Morfessor-VC", "3.3B", "arn-es")),
    (("Mono BPE", "3.3B", "arn-es"), ("Morfessor-VC", "3.3B", "arn-es")),
    (("Standard BPE", "600M", "es-arn"), ("Morfessor-VC", "600M", "es-arn")),
    (("Standard BPE", "1.3B", "es-arn"), ("Morfessor-VC", "1.3B", "es-arn")),
    (("Standard BPE", "3.3B", "es-arn"), ("Morfessor-VC", "3.3B", "es-arn")),
    (("Standard BPE", "3.3B", "es-arn"), ("Morfessor", "3.3B", "es-arn")),
]


def read_lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def stem_for(cond, size, d):
    if (cond, size, d) in V2_STEMS:
        return V2_PRED, V2_STEMS[(cond, size, d)]
    if cond in LLMS:
        return ARCHIVE, LLMS[cond].format(d=d)
    return ARCHIVE, STEMS[cond].format(size=size, d=d, nllb=NLLB[size])


def key(cond, size, d):
    return f"{cond}|{size}|{d}"


def main():
    chrf = CHRF(word_order=2)
    refs = {"arn-es": read_lines(TEST / "tgt.txt"), "es-arn": read_lines(TEST / "src.txt")}
    srcs = {"arn-es": read_lines(TEST / "src.txt"), "es-arn": read_lines(TEST / "tgt.txt")}
    assert all(len(v) == 9382 for v in list(refs.values()) + list(srcs.values()))

    systems = [(c, s, d) for c in STEMS for s in SIZES for d in ("arn-es", "es-arn")]
    systems += [(c, "8B", d) for c in LLMS for d in ("arn-es", "es-arn")]

    stats, scores, out = {}, {}, {}
    for cond, size, d in systems:
        base, stem = stem_for(cond, size, d)
        rows = json.loads((base / f"{stem}_results.json").read_text(encoding="utf-8"))
        preds = read_lines(base / f"{stem}_preds.txt")
        assert len(preds) == len(rows) == 9382, stem
        assert preds == [r["pred"] for r in rows], f"{stem}: preds.txt != results.json"
        st = np.array(chrf._extract_corpus_statistics(preds, [refs[d]]))
        sent = np.array([chrf._compute_score_from_stats(list(x)).score for x in st])
        stored = np.array([r["chrf"] for r in rows])
        # Stored per-sentence scores were computed by predict_test.py with the
        # same metric; they must agree (to rounding) with fresh scoring here.
        max_dev = float(np.abs(np.round(sent, 2) - stored).max())
        assert max_dev <= 0.011, f"{stem}: stored per-sentence chrF++ differs by {max_dev}"
        corpus = chrf._compute_score_from_stats(list(st.sum(axis=0))).score
        assert abs(corpus - chrf.corpus_score(preds, [refs[d]]).score) < 1e-9
        bleu = sacrebleu.corpus_bleu(preds, [refs[d]]).score
        k = key(cond, size, d)
        stats[k], scores[k] = st, sent
        out[k] = {
            "condition": cond, "size": size, "direction": d, "file": str(base / stem),
            "corpus_chrfpp": round(corpus, 2),
            "mean_sentence_chrfpp": round(float(sent.mean()), 2),
            "corpus_bleu": round(bleu, 2),
            "source": "v2 retrain" if (cond, size, d) in V2_STEMS else "original prediction file",
        }
        print(f"{k:40s} corpus {corpus:6.2f}  mean {sent.mean():6.2f}  BLEU {bleu:6.2f}  ({out[k]['source']})")

    # Paired bootstrap: same resampled indices for both statistics.
    def draws():
        # identical index sequence for every comparison and both statistics
        rng = np.random.default_rng(SEED)
        for _ in range(N_BOOT):
            yield rng.integers(0, 9382, 9382)

    corpus_of = lambda st: chrf._compute_score_from_stats(list(st)).score

    def summarise(obs, diffs):
        diffs = np.sort(np.array(diffs))
        lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        return {"diff": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                "p_two_sided": round(float(min(p, 1.0)), 4),
                "significant_0.05": bool(lo > 0 or hi < 0)}

    comps = {}
    for a, b in COMPARISONS:
        ka, kb = key(*a), key(*b)
        sa, sb, xa, xb = stats[ka], stats[kb], scores[ka], scores[kb]
        cdiffs, mdiffs = [], []
        for i in draws():
            cdiffs.append(corpus_of(sb[i].sum(axis=0)) - corpus_of(sa[i].sum(axis=0)))
            mdiffs.append(float(xb[i].mean() - xa[i].mean()))
        name = f"{kb} minus {ka}"
        comps[name] = {
            "a": ka, "b": kb,
            "corpus_chrfpp": summarise(corpus_of(sb.sum(axis=0)) - corpus_of(sa.sum(axis=0)), cdiffs),
            "mean_sentence_chrfpp": summarise(float(xb.mean() - xa.mean()), mdiffs),
        }
        c, m = comps[name]["corpus_chrfpp"], comps[name]["mean_sentence_chrfpp"]
        print(f"{name:75s} corpus {c['diff']:+.2f} {c['ci95']} p={c['p_two_sided']}  |  "
              f"mean {m['diff']:+.2f} {m['ci95']} p={m['p_two_sided']}")

    paper = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}
    paper["v2"] = {
        **paper.get("v2", {}),
        "scores": out,
        "comparisons": comps,
        "bootstrap": {"n": N_BOOT, "seed": SEED, "method": "paired, same resampled indices for "
                      "corpus chrF++ (recomputed from summed sufficient statistics) and mean "
                      "sentence chrF++; two-sided percentile p"},
        "test_set": {"n": 9382, "path": str(TEST)},
        "metric": "sacrebleu CHRF(word_order=2) [chrF++]; corpus_bleu defaults",
    }
    STATS.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged into {STATS}")


if __name__ == "__main__":
    main()
