#!/usr/bin/env python3
"""Record the current Hugging Face release state for README generation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
ALLOWED_METADATA = {
    ".gitattributes", "README.md", "added_tokens.json", "config.json",
    "generation_config.json", "inference.py", "mapudungun_morfessor_model.bin",
    "mapudungun_segmenter.model", "release_metadata.json", "sentencepiece.bpe.model",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
}
REQUIRED_FILES = {
    "README.md", "config.json", "inference.py", "release_metadata.json",
    "tokenizer_config.json",
}


def is_allowed_file(filename):
    name = Path(filename).name
    return (name in ALLOWED_METADATA
            or name.endswith(".safetensors")
            or name.endswith(".safetensors.index.json")
            or (name.startswith("pytorch_model") and name.endswith(".bin"))
            or name == "pytorch_model.bin.index.json")


def has_weights(files):
    return any(Path(filename).name.endswith(".safetensors")
               or (Path(filename).name.startswith("pytorch_model")
                   and Path(filename).name.endswith(".bin"))
               for filename in files)


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
    api = HfApi()
    collection = api.get_collection(matrix["collection"])
    actual = {item.item_id for item in collection.items if item.item_type == "model"}
    expected = expected_repos(matrix)
    repository_errors = []
    repositories = {}
    for size in matrix["sizes"]:
        for direction in matrix["directions"]:
            for condition, condition_spec in matrix["conditions"].items():
                repo_id = f"{matrix['organization']}/mapudungun-nllb-{size}-{direction}-{condition}"
                if repo_id not in actual:
                    continue
                files = sorted(api.list_repo_files(repo_id))
                basenames = {Path(filename).name for filename in files}
                missing = sorted(REQUIRED_FILES - basenames)
                if not has_weights(files):
                    missing.append("model weights")
                assets = condition_spec["asset"]
                if assets:
                    if isinstance(assets, str):
                        assets = [assets]
                    for asset in assets:
                        staged_name = ("mapudungun_morfessor_model.bin"
                                       if asset == "mapudungun_morfessor_model.bin"
                                       else "mapudungun_segmenter.model")
                        if staged_name not in basenames:
                            missing.append(staged_name)
                forbidden = sorted(filename for filename in files if not is_allowed_file(filename))
                if missing:
                    repository_errors.append(f"{repo_id}: missing {', '.join(sorted(set(missing)))}")
                if forbidden:
                    repository_errors.append(f"{repo_id}: forbidden files {', '.join(forbidden)}")
                repositories[repo_id] = {
                    "file_count": len(files),
                    "missing": sorted(set(missing)),
                    "forbidden": forbidden,
                }

    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "collection": matrix["collection"],
        "expected": sorted(expected),
        "released": sorted(actual & expected),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "repository_validation": {
            "validated": len(repositories),
            "errors": repository_errors,
            "repositories": repositories,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(payload['released'])}/{len(payload['expected'])} expected models released")
    print(f"Validated {len(repositories)} live repositories; errors={len(repository_errors)}")
    print(f"Wrote {args.output}")
    errors = []
    if payload["missing"]:
        errors.append("missing collection items: " + ", ".join(payload["missing"]))
    if payload["unexpected"]:
        errors.append("unexpected collection items: " + ", ".join(payload["unexpected"]))
    errors.extend(repository_errors)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
