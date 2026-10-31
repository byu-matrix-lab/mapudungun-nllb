"""v2 option (c): score the 256-token Standard vs Morfessor-VC reruns.

All 12 systems share training data (regenerated train/dev), max length (256),
segmentation model and test set (ORIGINAL 9,382 pairs), so Morfessor-VC minus
Standard is a like-for-like comparison. Also measures how often each input
format exceeds 128 / 256 NLLB tokens (the truncation bias the 256 runs remove).

Merges results into results/paper_stats.json under ["v2"]["len256"].
"""

import json
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from transformers import NllbTokenizer

REPO = Path(__file__).resolve().parents[2]
V2 = Path("/nobackup/archive/usr/it238/mapudungun_v2")
PRED = V2 / "predictions"
DATA = V2 / "data" / "blocks"
TEST = DATA / "cleaned" / "test" / "cleaned"
STATS = REPO / "results" / "paper_stats.json"
TOKENIZER = ("/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/"
             "1a07f7d195896b2114afcb79b7b57ab512e7b43e")
N_BOOT, SEED, N = 10_000, 42, 9382
SIZES = ["600M", "1.3B", "3.3B"]
TOKS = ["standard", "morfessor_vc"]


def read_lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main():
    chrf = CHRF(word_order=2)
    refs = {"arn-es": read_lines(TEST / "tgt.txt"), "es-arn": read_lines(TEST / "src.txt")}
    corpus_of = lambda st: chrf._compute_score_from_stats(list(st)).score

    out, stats, sent = {}, {}, {}
    for d in refs:
        for size in SIZES:
            for tok in TOKS:
                stem = f"v2c-{size}-{d}-{tok}-len256"
                rows = json.loads((PRED / f"{stem}_results.json").read_text(encoding="utf-8"))
                preds = read_lines(PRED / f"{stem}_preds.txt")
                assert len(preds) == len(rows) == N and preds == [r["pred"] for r in rows], stem
                assert [r["ref"] for r in rows] == refs[d], f"{stem}: reference mismatch"
                assert not any("@@" in p for p in preds), f"{stem}: undesegmented output"
                st = np.array(chrf._extract_corpus_statistics(preds, [refs[d]]))
                sc = np.array([chrf._compute_score_from_stats(list(x)).score for x in st])
                k = f"{tok}|{size}|{d}"
                stats[k], sent[k] = st, sc
                out[k] = {"file": str(PRED / stem),
                          "corpus_chrfpp": round(corpus_of(st.sum(axis=0)), 2),
                          "mean_sentence_chrfpp": round(float(sc.mean()), 2),
                          "corpus_bleu": round(sacrebleu.corpus_bleu(preds, [refs[d]]).score, 2)}
                print(f"{k:28s} corpus {out[k]['corpus_chrfpp']:6.2f}  mean {out[k]['mean_sentence_chrfpp']:6.2f}"
                      f"  BLEU {out[k]['corpus_bleu']:6.2f}")

    def summarise(obs, diffs):
        diffs = np.sort(np.array(diffs))
        lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1]
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        return {"diff": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                "p_two_sided": round(float(min(p, 1.0)), 4), "significant_0.05": bool(lo > 0 or hi < 0)}

    comps = {}
    for d in refs:
        for size in SIZES:
            a, b = f"standard|{size}|{d}", f"morfessor_vc|{size}|{d}"
            rng = np.random.default_rng(SEED)
            cd, md = [], []
            for _ in range(N_BOOT):
                i = rng.integers(0, N, N)
                cd.append(corpus_of(stats[b][i].sum(axis=0)) - corpus_of(stats[a][i].sum(axis=0)))
                md.append(float(sent[b][i].mean() - sent[a][i].mean()))
            name = f"{b} minus {a}"
            comps[name] = {
                "corpus_chrfpp": summarise(corpus_of(stats[b].sum(axis=0)) - corpus_of(stats[a].sum(axis=0)), cd),
                "mean_sentence_chrfpp": summarise(float(sent[b].mean() - sent[a].mean()), md),
            }
            c, m = comps[name]["corpus_chrfpp"], comps[name]["mean_sentence_chrfpp"]
            print(f"VC - Std {size} {d}: corpus {c['diff']:+.2f} {c['ci95']} p={c['p_two_sided']} | "
                  f"mean {m['diff']:+.2f} {m['ci95']} p={m['p_two_sided']}")

    # Truncation exposure: share of sentences longer than 128 / 256 NLLB tokens
    # (language tag and </s> included), raw vs Morfessor-VC-segmented Mapudungun.
    tok = NllbTokenizer.from_pretrained(TOKENIZER)
    tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    tok.src_lang = "arn_Latn"
    trunc = {}
    for split in ["train", "test"]:
        files = {"raw": DATA / "cleaned" / split / "cleaned" / "src.txt",
                 "morfessor_vc": DATA / "morfessor_vc" / split / "src.txt"}
        for fmt, f in files.items():
            lens = np.array([len(x) for x in tok(read_lines(f))["input_ids"]])
            trunc[f"{split}|{fmt}"] = {"n": int(len(lens)),
                                       "pct_over_128": round(100 * float((lens > 128).mean()), 2),
                                       "pct_over_256": round(100 * float((lens > 256).mean()), 2)}
            print(f"truncation {split:5s} {fmt:12s} >128: {trunc[f'{split}|{fmt}']['pct_over_128']:5.2f}%  "
                  f">256: {trunc[f'{split}|{fmt}']['pct_over_256']:5.2f}%")

    paper = json.loads(STATS.read_text(encoding="utf-8"))
    paper.setdefault("v2", {})["len256"] = {
        "scores": out, "comparisons": comps, "truncation_arn_side": trunc,
        "bootstrap": {"n": N_BOOT, "seed": SEED, "method": "paired; same indices for both statistics"},
        "setup": "Standard and Morfessor-VC, max length 256 (train + dev generation), regenerated "
                 "train/dev, one Morfessor model trained on regenerated train, ORIGINAL test set, "
                 "predict_test.py decoding (greedy, max_new_tokens 256)",
    }
    STATS.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged into {STATS}")


if __name__ == "__main__":
    main()
