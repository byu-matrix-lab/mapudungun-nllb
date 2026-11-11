"""Shared setup for the regression tests (stdlib unittest; no pytest needed).

Run from the repo root:  .venv/bin/python -m unittest discover -s tests -v

Tests that need the archived data or the HuggingFace cache skip when those are
not present, so the suite also runs on a machine without them.
"""

import os
import getpass
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[1]
for sub in ("training", "eval", "tokenization", "analysis"):
    path = str(REPO / "scripts" / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

ARCHIVE = Path("/nobackup/archive/usr") / getpass.getuser()
PREDICTIONS = ARCHIVE / "mapudungun_predictions"
V2_TEST = ARCHIVE / "mapudungun_v2/data/blocks/cleaned/test/cleaned"
REBUILD = ARCHIVE / "mapudungun_rebuild"
HF_HUB = Path.home() / ".cache/huggingface/hub"
NLLB_600M = HF_HUB / "models--facebook--nllb-200-distilled-600M"

HAVE_ARCHIVE = PREDICTIONS.is_dir() and V2_TEST.is_dir()
HAVE_NLLB = NLLB_600M.is_dir()


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def table_cells():
    """The 58 published table cells and the prediction file behind each."""
    import json
    return json.loads((REPO / "results/v2_phaseA_audit.json").read_text(encoding="utf-8"))["cells"]
