"""Did a pre-segmented condition actually train on segmented data?

finetune_nllb.py once fell back silently to the unsegmented data for any
tokenizer approach its loader did not list (the cause of the identical Optuna
trials). UnigramLM, Optuna BPE and Morfessor-BPE were trained on 2026-04-05,
a day before the commit that lists them. This script translates the first N
test sentences two ways, with the segmented input the paper used and with raw
input, using predict_test.py's decoding. A model trained on segmented text
should do better on segmented input; Joint-5K (trained 2026-03-16 under a
listed name) is the reference. The segmented run must also reproduce the
archived predictions, which checks the setup.

Writes results/v2_loader_check.json.
"""

import json
import sys
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from predict_test import load_tokenizer, translate_batch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/nobackup/archive/usr/it238")
PRED = ARCHIVE / "mapudungun_predictions"
TEST_SRC = ARCHIVE / "mapudungun_v2/data/blocks/cleaned/test/cleaned/src.txt"
N = 1000
BATCH = 16  # predict_test.py's batch size: padding changes greedy outputs slightly

MODELS = {
    "Joint-5K BPE 600M (reference)": ("mapudungun_final_models/nllb-600M-blocks-arn-es-joint_bpe",
                                      "600M-blocks-arn-es-joint_bpe"),
    "Optuna BPE 600M": ("mapudungun_final_models/nllb-600M-blocks-arn-es-optuna_bpe",
                        "nllb-600M-blocks-arn-es-optuna_bpe"),
    "UnigramLM 1.3B": ("mapudungun_models/nllb-1.3B-blocks-arn-es-unigram_lm/best",
                       "nllb-1.3B-blocks-arn-es-unigram_lm"),
    "Morfessor-BPE 1.3B": ("mapudungun_models/nllb-1.3B-blocks-arn-es-morfessor_bpe/best",
                           "nllb-1.3B-blocks-arn-es-morfessor_bpe"),
}


def main():
    chrf = CHRF(word_order=2)
    raw = TEST_SRC.read_text(encoding="utf-8").splitlines()[:N]
    tok = load_tokenizer()
    tgt_id = tok.convert_tokens_to_ids("spa_Latn")
    out = {}
    for name, (mdir, stem) in MODELS.items():
        rows = json.loads((PRED / f"{stem}_results.json").read_text(encoding="utf-8"))[:N]
        seg = [r["src"] for r in rows]
        refs = [r["ref"] for r in rows]
        archived = [r["pred"] for r in rows]
        model = AutoModelForSeq2SeqLM.from_pretrained(ARCHIVE / mdir, dtype=torch.float16).to("cuda").eval()
        res = {}
        for label, src in (("segmented", seg), ("raw", raw)):
            preds = []
            for i in range(0, N, BATCH):
                preds += translate_batch(model, tok, src[i:i + BATCH], "arn_Latn", tgt_id, "cuda")
            sent = [chrf.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
            res[label] = {"corpus_chrfpp": round(chrf.corpus_score(preds, [refs]).score, 2),
                          "mean_sentence_chrfpp": round(sum(sent) / N, 2)}
            if label == "segmented":
                res["segmented_matches_archived"] = sum(p == a for p, a in zip(preds, archived))
        res["segmented_minus_raw_corpus"] = round(res["segmented"]["corpus_chrfpp"] - res["raw"]["corpus_chrfpp"], 2)
        out[name] = res
        print(name, json.dumps(res), flush=True)
        del model
        torch.cuda.empty_cache()
    path = REPO / "results" / "v2_loader_check.json"
    path.write_text(json.dumps({"n_sentences": N, "models": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
