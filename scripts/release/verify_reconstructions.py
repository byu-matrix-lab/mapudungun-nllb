"""Are the rebuilt checkpoints identical to the lost originals?

Five released conditions lost their original checkpoints and were retrained with
the paper recipe on the exact original data (configs/slurm/rebuild_missing_*.slurm).
The originals' test predictions survive. This translates the test set with each
rebuild, using the exact inputs the original saw (the `src` field of its results
file) and the paper's decoding (scripts/eval/predict_test.py: batch 16, greedy,
max_new_tokens 256, fp16; es->arn output desegmented), and compares with the
archived predictions line by line.

Writes results/release_reconstruction_check.json.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from predict_test import BATCH_SIZE, DESEG_RE, load_tokenizer, translate_batch  # noqa: E402

ARCHIVE = Path("/nobackup/archive/usr/it238")
REBUILT = ARCHIVE / "mapudungun_release_models"
PRED = ARCHIVE / "mapudungun_predictions"
MODELS = {  # rebuilt directory -> archived prediction stem of the lost original
    "nllb-600M-blocks-arn-es-morfessor_bpe": "nllb-600M-blocks-arn-es-morfessor_bpe",
    "nllb-600M-blocks-es-arn-morfessor_bpe": "nllb-600M-blocks-es-arn-morfessor_bpe",
    "nllb-600M-blocks-arn-es-unigram_lm": "nllb-600M-blocks-arn-es-unigram_lm",
    "nllb-600M-blocks-es-arn-unigram_lm": "nllb-600M-blocks-es-arn-unigram_lm",
    "nllb-200-3.3B-blocks-es-arn-unigram_lm": "nllb-200-3.3B-blocks-es-arn-unigram_lm",
}
OUT = REPO / "results" / "release_reconstruction_check.json"
PRED_OUT = ARCHIVE / "mapudungun_release_predictions"


def mean_sentence_chrf(preds, refs):
    metric = CHRF(word_order=2)
    return sum(metric.sentence_score(pred, [ref]).score for pred, ref in zip(preds, refs)) / len(refs)


def text_sha256(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def main():
    tok = load_tokenizer()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for check in out.values():
        local_prediction_path = check.pop("rebuilt_predictions", None)
        if local_prediction_path:
            check["rebuilt_prediction_file"] = Path(local_prediction_path).name
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    for rebuilt, stem in MODELS.items():
        existing = out.get(rebuilt, {})
        if "rebuilt_mean_sentence_chrfpp" in existing:
            continue
        rows = json.loads((PRED / f"{stem}_results.json").read_text(encoding="utf-8"))
        archived = (PRED / f"{stem}_preds.txt").read_text(encoding="utf-8").splitlines()
        refs = [r["ref"] for r in rows]
        archived_mean = mean_sentence_chrf(archived, refs)
        if existing.get("all_identical"):
            existing.update({
                "archived_mean_sentence_chrfpp": archived_mean,
                "rebuilt_mean_sentence_chrfpp": archived_mean,
                "absolute_chrfpp_difference": 0.0,
                "archived_prediction_sha256": text_sha256(archived),
                "rebuilt_prediction_sha256": text_sha256(archived),
            })
            out[rebuilt] = existing
            OUT.write_text(json.dumps(out, indent=2) + "\n")
            continue
        srcs = [r["src"] for r in rows]
        es_arn = "-es-arn-" in rebuilt
        src_lang, tgt_lang = ("spa_Latn", "arn_Latn") if es_arn else ("arn_Latn", "spa_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(REBUILT / rebuilt / "best", dtype=torch.float16).to("cuda").eval()
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        preds = []
        for i in range(0, len(srcs), BATCH_SIZE):
            batch = translate_batch(model, tok, srcs[i:i + BATCH_SIZE], src_lang, tgt_id, "cuda")
            preds += [DESEG_RE.sub("", p) for p in batch] if es_arn else batch
        pred_path = PRED_OUT / f"{rebuilt}_preds.txt"
        pred_path.write_text("\n".join(preds) + "\n", encoding="utf-8")
        same = sum(a == b for a, b in zip(preds, archived))
        rebuilt_mean = mean_sentence_chrf(preds, refs)
        out[rebuilt] = {"archived": stem, "n": len(archived), "identical_lines": same,
                        "all_identical": same == len(archived) == len(preds),
                        "first_differences": [i for i, (a, b) in enumerate(zip(preds, archived)) if a != b][:20],
                        "archived_mean_sentence_chrfpp": archived_mean,
                        "rebuilt_mean_sentence_chrfpp": rebuilt_mean,
                        "absolute_chrfpp_difference": abs(rebuilt_mean - archived_mean),
                        "archived_prediction_sha256": text_sha256(archived),
                        "rebuilt_prediction_sha256": text_sha256(preds),
                        "rebuilt_prediction_file": pred_path.name}
        print(rebuilt, out[rebuilt]["identical_lines"], "/", len(archived), flush=True)
        OUT.write_text(json.dumps(out, indent=2))
        del model
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
