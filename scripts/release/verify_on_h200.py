"""Is the prediction mismatch of two rebuilt checkpoints an inference-hardware effect?

verify_reconstructions.py (run on an A100) found that two rebuilt checkpoints differ
from the lost originals on about 2% of test lines: 600M arn->es UnigramLM and 3.3B
es->arn UnigramLM. Those two originals' archived predictions were generated on an
H200 (eng-1-1, jobs 11231036 and 11237283); the three that matched were generated on
A100s. The 3.3B rebuild's surviving weight shard is byte-identical to the original's.
This repeats the comparison for the two models on an H200 with the same decoding.
If they match exactly, the rebuilt checkpoints are identical to the originals and
the earlier mismatch came from the GPU type used for inference.

Writes results/release_reconstruction_check_h200.json.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from predict_test import BATCH_SIZE, DESEG_RE, load_tokenizer, translate_batch  # noqa: E402

ARCHIVE = Path("/nobackup/archive/usr/it238")
REBUILT = ARCHIVE / "mapudungun_release_models"
PRED = ARCHIVE / "mapudungun_predictions"
MODELS = ["nllb-600M-blocks-arn-es-unigram_lm", "nllb-200-3.3B-blocks-es-arn-unigram_lm"]
OUT = REPO / "results" / "release_reconstruction_check_h200.json"


def main():
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu, flush=True)
    tok = load_tokenizer()
    out = {"gpu": gpu, "models": {}}
    for name in MODELS:
        rows = json.loads((PRED / f"{name}_results.json").read_text(encoding="utf-8"))
        archived = (PRED / f"{name}_preds.txt").read_text(encoding="utf-8").splitlines()
        es_arn = "-es-arn-" in name
        src_lang, tgt_lang = ("spa_Latn", "arn_Latn") if es_arn else ("arn_Latn", "spa_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(REBUILT / name / "best", dtype=torch.float16).to("cuda").eval()
        tgt_id = tok.convert_tokens_to_ids(tgt_lang)
        srcs = [r["src"] for r in rows]
        preds = []
        for i in range(0, len(srcs), BATCH_SIZE):
            batch = translate_batch(model, tok, srcs[i:i + BATCH_SIZE], src_lang, tgt_id, "cuda")
            preds += [DESEG_RE.sub("", p) for p in batch] if es_arn else batch
        same = sum(a == b for a, b in zip(preds, archived))
        out["models"][name] = {"n": len(archived), "identical_lines": same,
                               "all_identical": same == len(archived) == len(preds),
                               "first_differences": [i for i, (a, b) in enumerate(zip(preds, archived)) if a != b][:20]}
        print(name, same, "/", len(archived), flush=True)
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        del model
        torch.cuda.empty_cache()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
