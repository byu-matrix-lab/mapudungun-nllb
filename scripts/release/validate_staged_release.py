#!/usr/bin/env python3
"""Validate self-contained Hugging Face staging directories against the release matrix."""

import argparse
import getpass
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = Path("/nobackup/archive/usr") / getpass.getuser() / "mapudungun_hf_stage"
FORBIDDEN = {"optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--matrix", type=Path, default=REPO / "release/model_matrix.json")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO / "results/staged_release_validation.json")
    return parser.parse_args()


def has_weights(path):
    return bool(list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin")))


def main():
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    expected = []
    errors = []
    packages = {}
    for size in matrix["sizes"]:
        for direction in matrix["directions"]:
            for condition, condition_spec in matrix["conditions"].items():
                name = f"mapudungun-nllb-{size}-{direction}-{condition}"
                expected.append(name)
                path = args.stage_dir / name
                if not path.is_dir():
                    continue
                missing = [filename for filename in
                           ("README.md", "config.json", "tokenizer_config.json", "inference.py",
                            "release_metadata.json") if not (path / filename).is_file()]
                if not has_weights(path):
                    missing.append("model weights")
                assets = condition_spec["asset"]
                if assets:
                    if isinstance(assets, str):
                        assets = [assets]
                    for asset in assets:
                        staged_name = ("mapudungun_morfessor_model.bin"
                                       if asset == "mapudungun_morfessor_model.bin"
                                       else "mapudungun_segmenter.model")
                        if not (path / staged_name).is_file():
                            missing.append(staged_name)
                forbidden = sorted(item.name for item in path.iterdir() if item.name in FORBIDDEN)
                if missing:
                    errors.append(f"{name}: missing {', '.join(sorted(set(missing)))}")
                if forbidden:
                    errors.append(f"{name}: forbidden training files {', '.join(forbidden)}")
                if (path / "release_metadata.json").is_file():
                    metadata = json.loads((path / "release_metadata.json").read_text(encoding="utf-8"))
                    wanted = {"size": size, "direction": direction, "condition": condition}
                    mismatched = {key: (metadata.get(key), value) for key, value in wanted.items()
                                  if metadata.get(key) != value}
                    if mismatched:
                        errors.append(f"{name}: metadata mismatch {mismatched}")
                    if metadata.get("checkpoint_provenance") == "reconstructed release run":
                        check = metadata.get("reconstruction_verification", {})
                        for field in ("n", "identical_lines", "all_identical",
                                      "archived_mean_sentence_chrfpp", "rebuilt_mean_sentence_chrfpp",
                                      "archived_prediction_sha256", "rebuilt_prediction_sha256"):
                            if field not in check:
                                errors.append(f"{name}: reconstruction verification missing {field}")
                packages[name] = {
                    "file_count": sum(1 for item in path.iterdir() if item.is_file()),
                    "weight_files": sorted(item.name for item in path.glob("*.safetensors"))
                                    + sorted(item.name for item in path.glob("pytorch_model*.bin")),
                }

    missing_packages = sorted(set(expected) - packages.keys())
    unexpected_packages = sorted(item.name for item in args.stage_dir.iterdir()
                                 if item.is_dir() and item.name not in expected)
    if missing_packages and not args.allow_partial:
        errors.append("missing packages: " + ", ".join(missing_packages))
    if unexpected_packages:
        errors.append("unexpected packages: " + ", ".join(unexpected_packages))
    result = {
        "expected_packages": len(expected),
        "validated_packages": len(packages),
        "missing_packages": missing_packages,
        "unexpected_packages": unexpected_packages,
        "errors": errors,
        "packages": packages,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(packages)}/{len(expected)} packages; errors={len(errors)}")
    print(f"wrote {args.output}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
