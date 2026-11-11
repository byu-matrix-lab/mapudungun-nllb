#!/usr/bin/env python3
"""Record the current Hugging Face release state for README generation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]


def expected_repos(matrix):
    return {
        f"{matrix['organization']}/mapudungun-nllb-{size}-{direction}-{condition}"
        for size in matrix["sizes"]
        for direction in matrix["directions"]
        for condition in matrix["conditions"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=REPO / "release/model_matrix.json")
    parser.add_argument("--output", type=Path, default=REPO / "results/release_status.json")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    collection = HfApi().get_collection(matrix["collection"])
    actual = {item.item_id for item in collection.items if item.item_type == "model"}
    expected = expected_repos(matrix)
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "collection": matrix["collection"],
        "expected": sorted(expected),
        "released": sorted(actual & expected),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(payload['released'])}/{len(payload['expected'])} expected models released")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
