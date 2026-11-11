#!/usr/bin/env python3
"""Stage and optionally publish the complete Mapudungun NLLB model matrix.

Default mode is read-only. Use --stage to build self-contained upload
directories and --upload to publish them. Source checkpoints are never changed.
"""

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/nobackup/archive/usr") / getpass.getuser()
MATRIX_PATH = REPO / "release/model_matrix.json"
STATS_PATH = REPO / "results/paper_stats.json"
INFERENCE = REPO / "scripts/model_inference.py"

PAPER_CITE = """@inproceedings{thompson2026mapudungun,
  title     = {Bringing {Mapudungun} into the Modern {MT} Ecosystem: Morphology-Aware Tokenization for {NLLB}-200 Fine-Tuning},
  author    = {Thompson, Isaac M. and Rogers, Brandon M.A. and Ringger, Eric K.},
  booktitle = {Proceedings of the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)},
  year      = {2026}
}"""

DESCRIPTIONS = {
    "standard-bpe": "NLLB-200's built-in SentencePiece tokenizer, with no external presegmentation.",
    "joint-bpe": "A joint Mapudungun-Spanish SentencePiece BPE condition.",
    "mono-bpe": "A Mapudungun-only SentencePiece BPE condition.",
    "optuna-bpe": "A Mapudungun-only SentencePiece BPE condition with vocabulary size selected by the paper's Optuna study.",
    "morfessor": "Unsupervised Morfessor segmentation applied to Mapudungun.",
    "morfessor-vc": "Morfessor segmentation followed by greedy removal of boundaries whose merged form is an existing NLLB vocabulary item.",
    "morfessor-bpe": "Morfessor segmentation followed by SentencePiece BPE within morphemes.",
    "unigram-lm": "A Mapudungun-only SentencePiece UnigramLM condition.",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--final-root", type=Path, default=ARCHIVE / "mapudungun_final_models")
    parser.add_argument("--training-root", type=Path, default=ARCHIVE / "mapudungun_models")
    parser.add_argument("--corrected-root", type=Path, default=ARCHIVE / "mapudungun_v2/models")
    parser.add_argument("--reconstructed-root", type=Path, default=ARCHIVE / "mapudungun_release_models")
    parser.add_argument("--asset-root", type=Path, default=ARCHIVE / "mapudungun_bpe_models")
    parser.add_argument("--morfessor-model", type=Path, default=ARCHIVE / "mapudungun_morfessor_model.bin")
    parser.add_argument("--stage-dir", type=Path, default=ARCHIVE / "mapudungun_hf_stage")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--condition")
    parser.add_argument("--size")
    parser.add_argument("--direction")
    return parser.parse_args()


def archive_name(matrix, size, direction, condition):
    spec = matrix["conditions"][condition]
    return matrix["sizes"][size]["archive_prefix"] + "-" + direction + spec["archive_suffix"]


def model_weight_exists(path):
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def validate_model(path):
    missing = [name for name in ("config.json", "tokenizer_config.json") if not (path / name).is_file()]
    if not model_weight_exists(path):
        missing.append("model weights")
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")


def best_checkpoint(training_dir):
    best = training_dir / "best"
    if best.is_dir():
        return best
    candidates = []
    for state_path in training_dir.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        selected = state.get("best_model_checkpoint")
        if selected:
            candidate = Path(selected)
            local_candidate = training_dir / candidate.name
            if local_candidate.is_dir():
                candidate = local_candidate
            elif not candidate.is_absolute():
                candidate = local_candidate
            if candidate.is_dir():
                candidates.append(candidate)
    unique = {path.resolve() for path in candidates}
    return unique.pop() if len(unique) == 1 else None


def resolve_source(args, matrix, size, direction, condition):
    name = archive_name(matrix, size, direction, condition)
    if condition == "standard-bpe" and direction == "arn-es":
        corrected = args.corrected_root / f"{matrix['sizes'][size]['archive_prefix']}-{direction}-origdata" / "best"
        if corrected.is_dir():
            return corrected, "corrected original-data retrain"
    reconstructed = args.reconstructed_root / name
    for candidate in (reconstructed / "best", reconstructed):
        if candidate.is_dir() and model_weight_exists(candidate):
            return candidate, "reconstructed release run"
    final = args.final_root / name
    if final.is_dir():
        try:
            validate_model(final)
            return final, "paper-era archived checkpoint"
        except ValueError:
            pass
    training = args.training_root / name
    if training.is_dir():
        selected = best_checkpoint(training)
        if selected:
            return selected, "paper-era training checkpoint"
    return None, None


def asset_paths(args, condition_spec):
    names = condition_spec["asset"]
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    paths = []
    for name in names:
        path = args.morfessor_model if name == "mapudungun_morfessor_model.bin" else args.asset_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append((name, path))
    return paths


def make_card(matrix, stats, size, direction, condition, repo_id, provenance):
    size_spec = matrix["sizes"][size]
    direction_spec = matrix["directions"][direction]
    condition_spec = matrix["conditions"][condition]
    result = stats["cells"][f"{condition_spec['label']}|{size}|{direction}"]["chrfpp_mean"]
    return f"""---
language:
- arn
- es
license: other
license_name: NLLB and corpus-derived terms
pipeline_tag: translation
base_model: {size_spec['base_model']}
tags:
- nllb
- mapudungun
- spanish
- low-resource-mt
- morphology
metrics:
- chrf
---

# {condition_spec['label']}: {size}, {direction_spec['label']}

Fine-tuned {size_spec['base_model']} checkpoint for {direction_spec['label']}.

{DESCRIPTIONS[condition]}

The corrected paper pipeline reports mean sentence-level chrF++ **{result:.2f}** for this condition, size, and direction. This card was generated from the repository's results/paper_stats.json source of truth.

Checkpoint provenance: {provenance}.

## Inference

Download this repository and run:

    python inference.py --model {repo_id} "text to translate"

The helper applies condition-specific Mapudungun preprocessing and removes @@ continuation markers from Mapudungun output. Transformers 4.x is required.

## License and data

The base model is NLLB-200 and is distributed under CC-BY-NC 4.0. The fine-tuning corpus is derived from the Duan et al. Mapudungun corpus, distributed under CC BY-NC-SA 3.0. The card uses license: other so neither upstream obligation is hidden; users must comply with both sources' terms.

## Citation

    {PAPER_CITE.replace(chr(10), chr(10) + '    ')}
"""


def copy_asset(stage, name, source):
    target_name = name if name == "mapudungun_morfessor_model.bin" else "mapudungun_segmenter.model"
    shutil.copy2(source, stage / target_name)


def stage_model(args, matrix, stats, size, direction, condition, source, provenance):
    repo_id = f"{matrix['organization']}/mapudungun-nllb-{size}-{direction}-{condition}"
    destination = args.stage_dir / repo_id.split("/", 1)[1]
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    excluded = {"README.md", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    for item in source.iterdir():
        if item.name not in excluded and item.is_file():
            shutil.copy2(item, destination / item.name)
    validate_model(destination)
    for name, path in asset_paths(args, matrix["conditions"][condition]):
        copy_asset(destination, name, path)
    shutil.copy2(INFERENCE, destination / "inference.py")
    metadata = {
        "condition": condition,
        "condition_label": matrix["conditions"][condition]["label"],
        "size": size,
        "direction": direction,
        "source_lang": matrix["directions"][direction]["source"],
        "target_lang": matrix["directions"][direction]["target"],
        "base_model": matrix["sizes"][size]["base_model"],
        "checkpoint_provenance": provenance,
    }
    (destination / "release_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        make_card(matrix, stats, size, direction, condition, repo_id, provenance), encoding="utf-8")
    return repo_id, destination


def selections(args, matrix):
    for size in matrix["sizes"]:
        if args.size and args.size != size:
            continue
        for direction in matrix["directions"]:
            if args.direction and args.direction != direction:
                continue
            for condition in matrix["conditions"]:
                if args.condition and args.condition != condition:
                    continue
                yield size, direction, condition


def main():
    args = parse_args()
    if args.upload:
        args.stage = True
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))["paper_v2"]
    resolved, errors = [], []
    for size, direction, condition in selections(args, matrix):
        source, provenance = resolve_source(args, matrix, size, direction, condition)
        repo_id = f"{matrix['organization']}/mapudungun-nllb-{size}-{direction}-{condition}"
        if source is None:
            errors.append(f"{repo_id}: checkpoint unavailable")
            continue
        try:
            validate_model(source)
            asset_paths(args, matrix["conditions"][condition])
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
            continue
        resolved.append((size, direction, condition, source, provenance))
        print(f"OK {repo_id} <- {source}")

    if errors:
        print("\n".join(f"ERROR {error}" for error in errors), file=sys.stderr)
        if not args.allow_partial:
            raise SystemExit(f"{len(errors)} requested releases are not ready")

    staged = []
    if args.stage:
        for values in resolved:
            staged.append(stage_model(args, matrix, stats, *values))
        print(f"Staged {len(staged)} models under {args.stage_dir}")

    if args.upload:
        api = HfApi()
        for repo_id, directory in staged:
            create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
            upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="model",
                          commit_message="Publish validated Mapudungun NLLB release")
            api.add_collection_item(collection_slug=matrix["collection"], item_id=repo_id,
                                    item_type="model", exists_ok=True)
            print(f"Uploaded {repo_id}")


if __name__ == "__main__":
    main()
