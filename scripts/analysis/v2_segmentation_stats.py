"""Segmentation statistics for the v2 paper, from the original data and models.

Rebuilds each pre-segmented condition's Mapudungun segmentation of the recovered
original data with the segmentation model the paper used, and refuses to report
anything unless the rebuild reproduces the paper's stored test inputs exactly
(the `src` field of each condition's results file). Then computes, on train:
fertility (pre-segmented tokens per whitespace word), Morfessor boundaries and
the boundaries Morfessor-VC removes; and the share of sentences longer than 128
NLLB tokens (truncated in training) for raw and Morfessor-VC input.

Merges into results/paper_stats.json["paper_v2"]["segmentation"].
"""

import json
import sys
from pathlib import Path

import morfessor
import sentencepiece as spm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tokenization"))
import segment_bpe as sb  # noqa: E402
import segment_morfessor as sm  # noqa: E402
import segment_morfessor_bpe as smb  # noqa: E402
import segment_morfessor_vc as vc  # noqa: E402

ARCHIVE = Path("/nobackup/archive/usr/it238")
DATA = ARCHIVE / "mapudungun_rebuild/data-processed/blocks/cleaned"
PRED = ARCHIVE / "mapudungun_predictions"
BPE = ARCHIVE / "mapudungun_bpe_models"
MORFESSOR_MODEL = ARCHIVE / "mapudungun_morfessor_model.bin"
NLLB_SP = (Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/"
           "f8d333a098d19b4fd9a8b18f94170487ad3f821d/sentencepiece.bpe.model")
NLLB_TOKENIZER = (Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/"
                  "1a07f7d195896b2114afcb79b7b57ab512e7b43e")
STATS = REPO / "results" / "paper_stats.json"

# condition -> (stored-results stem whose `src` is the paper's test input, SentencePiece model or None)
SP_CONDITIONS = {
    "Joint-5K BPE": ("600M-blocks-arn-es-joint_bpe", "blocks_duan_bpe.model"),
    "Mono BPE": ("600M-blocks-arn-es-mono_bpe", "blocks_large_bpe_1000.model"),
    "Optuna BPE": ("nllb-600M-blocks-arn-es-optuna_bpe", "blocks_mono_bpe_5829.model"),
    "UnigramLM": ("nllb-600M-blocks-arn-es-unigram_lm", "blocks_unigram_lm_1000.model"),
}
MORFESSOR_STEM = "600M-arn-es-morfessor"
VC_STEM = "600M-blocks-arn-es-morfessor_vc"
MORF_BPE = ("nllb-600M-blocks-arn-es-morfessor_bpe", "blocks_morfessor_bpe_4000.model")


def read(split):
    return (DATA / split / "cleaned" / "src.txt").read_text(encoding="utf-8").splitlines()


def stored_src(stem):
    return [r["src"] for r in json.loads((PRED / f"{stem}_results.json").read_text(encoding="utf-8"))]


def load_sp(name):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(BPE / name))
    return sp


def fertility(segmented, raw):
    return sum(len(s.split()) for s in segmented) / sum(len(r.split()) for r in raw)


def main():
    train, test = read("train"), read("test")
    seg = {}  # condition -> {"train": [...], "test": [...]}

    model = morfessor.MorfessorIO().read_binary_model_file(str(MORFESSOR_MODEL))
    seg["Morfessor"] = {s: [sm.segment_line(l, model).rstrip("\n") for l in lines]
                        for s, lines in (("train", train), ("test", test))}
    nllb = spm.SentencePieceProcessor()
    nllb.Load(str(NLLB_SP))
    vocab = vc.build_vocab_set(nllb)
    seg["Morfessor-VC"] = {s: [vc.cascade_line(l, vocab).rstrip("\n") for l in seg["Morfessor"][s]]
                           for s in ("train", "test")}
    mb = load_sp(MORF_BPE[1])
    seg["Morfessor-BPE"] = {s: [smb.segment_line(l, mb).rstrip("\n") for l in seg["Morfessor"][s]]
                            for s in ("train", "test")}
    for cond, (_, model_name) in SP_CONDITIONS.items():
        sp = load_sp(model_name)
        seg[cond] = {s: [sb.segment_line(sp, l).rstrip("\n") for l in lines]
                     for s, lines in (("train", train), ("test", test))}

    stems = {"Morfessor": MORFESSOR_STEM, "Morfessor-VC": VC_STEM, "Morfessor-BPE": MORF_BPE[0],
             **{c: v[0] for c, v in SP_CONDITIONS.items()}}
    out = {"conditions": {}}
    for cond, stem in stems.items():
        ref = stored_src(stem)
        match = sum(a == b for a, b in zip(seg[cond]["test"], ref))
        out["conditions"][cond] = {
            "test_input_reproduced": match, "n_test": len(ref),
            "fertility_train": round(fertility(seg[cond]["train"], train), 3),
        }
        print(f"{cond:14s} reproduces {match}/{len(ref)} stored test inputs; "
              f"train fertility {out['conditions'][cond]['fertility_train']}", flush=True)
        assert match == len(ref), f"{cond}: rebuild does not reproduce the paper's test input"

    morf_b = sum(vc.boundary_count(l) for l in seg["Morfessor"]["train"])
    vc_b = sum(vc.boundary_count(l) for l in seg["Morfessor-VC"]["train"])
    out["boundaries_train"] = {"morfessor": morf_b, "morfessor_vc": vc_b, "removed": morf_b - vc_b,
                               "removed_pct": round(100 * (morf_b - vc_b) / morf_b, 1)}
    print("boundaries (train):", out["boundaries_train"])

    from transformers import NllbTokenizer
    tok = NllbTokenizer.from_pretrained(str(NLLB_TOKENIZER))
    tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    tok.src_lang = "arn_Latn"
    trunc = {}
    for name, lines in (("train_raw", train), ("train_morfessor_vc", seg["Morfessor-VC"]["train"]),
                        ("test_raw", test), ("test_morfessor_vc", seg["Morfessor-VC"]["test"])):
        lens = [len(x) for x in tok(lines)["input_ids"]]
        trunc[name] = {"pct_over_128": round(100 * sum(n > 128 for n in lens) / len(lens), 2),
                       "pct_over_256": round(100 * sum(n > 256 for n in lens) / len(lens), 2)}
    out["truncation"] = trunc
    print("truncation:", trunc)

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    stats.setdefault("paper_v2", {})["segmentation"] = out
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged into {STATS}")


if __name__ == "__main__":
    main()
