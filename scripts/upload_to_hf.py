"""
Archive best/ checkpoints and upload all 48 paper models to HuggingFace.

Usage:
    python scripts/upload_to_hf.py [--dry-run] [--skip-archive] [--skip-upload]

Each model goes to: byumatrixlab/mapudungun-nllb-{size}-{direction}-{condition}
All models are added to collection: byumatrixlab/mapudungun-nllb-6a0b99ce4bbdd3b46531c6b4
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

AUTODELETE_BASE = Path.home() / "nobackup/autodelete/mapudungun/models"
ARCHIVE_BASE = Path("/nobackup/archive/usr/it238/mapudungun_final_models")
COLLECTION_SLUG = "byumatrixlab/mapudungun-nllb-6a0b99ce4bbdd3b46531c6b4"
ORG = "byumatrixlab"

PAPER_CITE = """\
@inproceedings{thompson2026mapudungun,
  title     = {Bringing {Mapudungun} into the Modern {MT} Ecosystem: Morphology-Aware Tokenization for {NLLB}-200 Fine-Tuning},
  author    = {Thompson, Isaac},
  booktitle = {Proceedings of the 5th Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP 2026)},
  year      = {2026},
}"""

CONDITIONS = [
    ("",            "standard-bpe",  "Standard BPE",   "NLLB-200's default SentencePiece BPE, no modification."),
    ("-joint_bpe",  "joint-bpe",     "Joint-5K BPE",   "Joint Mapudungun+Spanish BPE with 5K merge operations (Duan et al. 2020)."),
    ("-mono_bpe",   "mono-bpe",      "Mono BPE",       "Monolingual Mapudungun BPE using the Gowda & May (2020) 95%-coverage vocabulary heuristic."),
    ("-optuna_bpe", "optuna-bpe",    "Optuna BPE",     "Monolingual BPE with vocabulary size tuned via Optuna hyperparameter search."),
    ("-morfessor",  "morfessor",     "Morfessor",      "Unsupervised morphological segmentation with Morfessor 2.0 (Virpioja et al. 2013) pre-applied to Mapudungun tokens."),
    ("-morfessor_vc","morfessor-vc", "Morfessor-VC",   "Morfessor-VC (this paper): Morfessor segmentation constrained to NLLB's pretrained SentencePiece vocabulary, eliminating double-tokenization."),
    ("-morfessor_bpe","morfessor-bpe","Morfessor-BPE", "BPE trained on Morfessor-segmented Mapudungun text (Banerjee & Bhattacharyya 2018)."),
    ("-unigram_lm", "unigram-lm",   "UnigramLM",      "SentencePiece UnigramLM trained on Mapudungun with the Gowda & May vocabulary size heuristic (Kudo 2018)."),
]

SIZES = [
    ("600M",    "nllb-600M-blocks",     "NLLB-200 distilled 600M"),
    ("1.3B",    "nllb-1.3B-blocks",     "NLLB-200 distilled 1.3B"),
    ("3.3B",    "nllb-200-3.3B-blocks", "NLLB-200 3.3B"),
]

DIRECTIONS = [
    ("arn-es", "arn_Latn", "spa_Latn", "Mapudungun→Spanish"),
    ("es-arn", "spa_Latn", "arn_Latn", "Spanish→Mapudungun"),
]


def make_model_card(condition_name, condition_desc, size_label, direction_label,
                    src_lang, tgt_lang, repo_id):
    return f"""\
---
language:
- arn
- es
license: cc-by-4.0
tags:
- translation
- nllb
- mapudungun
- low-resource-mt
- morphology
base_model: facebook/nllb-200-{size_label.lower().replace(" ", "-").replace("nllb-200 ", "")}
---

# {repo_id.split("/")[-1]}

Fine-tuned [{size_label}](https://huggingface.co/facebook/nllb-200-distilled-600M) for {direction_label} translation using the **{condition_name}** tokenization condition.

{condition_desc}

Part of the paper: *Bringing Mapudungun into the Modern MT Ecosystem: Morphology-Aware Tokenization for NLLB-200 Fine-Tuning* (AmericasNLP 2026 @ ACL).

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "translation",
    model="{repo_id}",
    src_lang="{src_lang}",
    tgt_lang="{tgt_lang}",
)
print(pipe("your text here", max_length=256))
```

## Citation

```bibtex
{PAPER_CITE}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--condition", help="Only process this condition slug (e.g. morfessor-vc)")
    parser.add_argument("--size", help="Only process this size (e.g. 600M)")
    parser.add_argument("--direction", help="Only process this direction (e.g. arn-es)")
    args = parser.parse_args()

    api = HfApi()

    errors = []

    for cond_suffix, cond_slug, cond_name, cond_desc in CONDITIONS:
        if args.condition and cond_slug != args.condition:
            continue
        for size_slug, dir_prefix, size_label in SIZES:
            if args.size and size_slug != args.size:
                continue
            for dir_slug, src_lang, tgt_lang, dir_label in DIRECTIONS:
                if args.direction and dir_slug != args.direction:
                    continue

                src_dir = AUTODELETE_BASE / f"{dir_prefix}-{dir_slug}{cond_suffix}" / "best"
                archive_dir = ARCHIVE_BASE / f"{dir_prefix}-{dir_slug}{cond_suffix}"
                repo_id = f"{ORG}/mapudungun-nllb-{size_slug}-{dir_slug}-{cond_slug}"

                print(f"\n{'='*60}")
                print(f"Model: {repo_id}")

                # --- Archive ---
                if not args.skip_archive:
                    if not src_dir.exists():
                        print(f"  [ARCHIVE] MISSING source: {src_dir}")
                        errors.append(f"missing source: {src_dir}")
                        continue

                    if archive_dir.exists():
                        print(f"  [ARCHIVE] already exists, skipping copy")
                    else:
                        print(f"  [ARCHIVE] {src_dir} -> {archive_dir}")
                        if not args.dry_run:
                            shutil.copytree(src_dir, archive_dir)
                            print(f"  [ARCHIVE] done")

                # --- Upload ---
                if not args.skip_upload:
                    upload_src = archive_dir if not args.skip_archive else src_dir
                    if not upload_src.exists():
                        print(f"  [UPLOAD] source not found: {upload_src}")
                        errors.append(f"upload source missing: {upload_src}")
                        continue

                    # Write model card
                    card_path = upload_src / "README.md"
                    card_text = make_model_card(
                        cond_name, cond_desc, size_label, dir_label,
                        src_lang, tgt_lang, repo_id
                    )
                    if not args.dry_run:
                        card_path.write_text(card_text)

                    # Create repo
                    print(f"  [UPLOAD] creating/verifying repo {repo_id}")
                    if not args.dry_run:
                        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

                    # Upload files
                    print(f"  [UPLOAD] uploading from {upload_src}")
                    if not args.dry_run:
                        upload_folder(
                            folder_path=str(upload_src),
                            repo_id=repo_id,
                            repo_type="model",
                            commit_message=f"Add {cond_name} {size_slug} {dir_label} model",
                            ignore_patterns=["training_args.bin"],
                        )
                        print(f"  [UPLOAD] done")

                    # Add to collection
                    print(f"  [COLLECTION] adding to {COLLECTION_SLUG}")
                    if not args.dry_run:
                        try:
                            api.add_collection_item(
                                collection_slug=COLLECTION_SLUG,
                                item_id=repo_id,
                                item_type="model",
                                exists_ok=True,
                            )
                            print(f"  [COLLECTION] done")
                        except Exception as e:
                            print(f"  [COLLECTION] warning: {e}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\nAll done.")


if __name__ == "__main__":
    main()
