"""COMET (wmt22-comet-da) at full precision for every system in the paper's tables.

comet_score.py stored system scores rounded to 4 decimals, so a value like 0.4885
cannot be rounded to 3 decimals reliably (the published tables round such ties
both ways). This rescoring keeps full precision. Every system gets the raw,
unsegmented source (pre-segmented arn->es conditions used to get their @@-marked
input). Predictions and references are unchanged.

Writes results/v2_comet_full.json: {stem: comet}.
"""

import json
from pathlib import Path

from comet import load_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/nobackup/archive/usr/it238")
PRED = ARCHIVE / "mapudungun_predictions"
V2_PRED = ARCHIVE / "mapudungun_v2/predictions"
TEST = ARCHIVE / "mapudungun_rebuild/data-processed/blocks/cleaned/test/cleaned"
CKPT = ("/home/it238/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/"
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")
OUT = REPO / "results" / "v2_comet_full.json"


def main():
    raw = {"arn-es": (TEST / "src.txt").read_text(encoding="utf-8").splitlines(),
           "es-arn": (TEST / "tgt.txt").read_text(encoding="utf-8").splitlines()}
    cells = json.loads((REPO / "results/v2_phaseA_audit.json").read_text())["cells"]
    jobs = [(PRED, c["file"], c["direction"]) for c in cells]
    jobs += [(V2_PRED, f"v2o-{s}-arn-es", "arn-es") for s in ("600M", "1.3B", "3.3B")]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    model = load_from_checkpoint(CKPT)
    for base, stem, d in jobs:
        if stem in out:
            continue
        rows = json.loads((base / f"{stem}_results.json").read_text(encoding="utf-8"))
        assert len(rows) == len(raw[d])
        data = [{"src": s, "mt": r["pred"], "ref": r["ref"]} for s, r in zip(raw[d], rows)]
        out[stem] = model.predict(data, batch_size=64, gpus=1).system_score
        OUT.write_text(json.dumps(out, indent=2))  # checkpoint after each system
        print(stem, out[stem], flush=True)
    print(f"wrote {OUT} ({len(out)} systems)")


if __name__ == "__main__":
    main()
