"""Rescore COMET (wmt22-comet-da) for pre-segmented arn->es conditions with the
raw source sentence. comet_score.py took `src` from each results file, and the
pre-segmented conditions store their @@-marked model input there, so the
published COMET appendix scored those rows against marked-up sources.

Predictions and references are unchanged. Writes results/v2_comet_clean_src.json.
"""

import json
from pathlib import Path

from comet import load_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
PRED = Path("/nobackup/archive/usr/it238/mapudungun_predictions")
RAW_SRC = Path("/nobackup/archive/usr/it238/mapudungun_v2/data/blocks/cleaned/test/cleaned/src.txt")
CKPT = ("/home/it238/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/"
        "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")


def main():
    cells = json.loads((REPO / "results" / "v2_phaseA_audit.json").read_text(encoding="utf-8"))["cells"]
    raw = RAW_SRC.read_text(encoding="utf-8").splitlines()
    model = load_from_checkpoint(CKPT)
    out = {}
    for c in cells:
        if c["direction"] != "arn-es":
            continue
        rows = json.loads((PRED / f"{c['file']}_results.json").read_text(encoding="utf-8"))
        if not any("@@" in r["src"] for r in rows):
            continue  # unsegmented input: published COMET already used the raw source
        assert len(rows) == len(raw)
        data = [{"src": s, "mt": r["pred"], "ref": r["ref"]} for s, r in zip(raw, rows)]
        score = model.predict(data, batch_size=64, gpus=1).system_score
        published = json.loads((PRED / f"{c['file']}_comet.json").read_text())["comet"]
        out[c["file"]] = {"condition": c["condition"], "size": c["size"],
                          "comet_raw_src": round(score, 4), "comet_published_marked_src": published}
        print(c["condition"], c["size"], out[c["file"]], flush=True)
    path = REPO / "results" / "v2_comet_clean_src.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
