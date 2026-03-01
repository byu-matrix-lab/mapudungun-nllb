#!/usr/bin/env python3
"""
finetune_nllb.py

Fine-tune facebook/nllb-200-distilled-600M (or -1.3B) bidirectionally on the
Mapudungun–Spanish parallel corpus.

Mapudungun (arn_Latn) is absent from NLLB-200's original 202 languages. We add
it as a new language token and initialize its embedding from the mean of the three
closest indigenous Latin-script languages present in NLLB: ayr_Latn (Aymara),
grn_Latn (Guaraní), quy_Latn (Quechua).

Usage — run once per direction:
  python scripts/training/finetune_nllb.py \\
      --approach lines --src arn --tgt es \\
      --data-dir /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --output-dir /home/it238/nobackup/autodelete/mapudungun/models/nllb-600M-lines-arn-es

  python scripts/training/finetune_nllb.py \\
      --approach lines --src es --tgt arn \\
      --data-dir /home/it238/nobackup/autodelete/mapudungun/data-processed \\
      --output-dir /home/it238/nobackup/autodelete/mapudungun/models/nllb-600M-lines-es-arn
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import sacrebleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language code constants
LANG_CODES = {"arn": "arn_Latn", "es": "spa_Latn"}
PROXY_LANGS = ["ayr_Latn", "grn_Latn", "quy_Latn"]  # for arn_Latn initialization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune NLLB-200 on Mapudungun–Spanish."
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--approach",
        choices=["lines", "blocks"],
        required=True,
        help="Extraction approach: 'lines' (~170K pairs) or 'blocks' (~55K pairs).",
    )
    parser.add_argument(
        "--src", choices=["arn", "es"], required=True, help="Source language."
    )
    parser.add_argument(
        "--tgt", choices=["arn", "es"], required=True, help="Target language."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root of data-processed directory (contains lines/ and blocks/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save model checkpoints and final model.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=4, help="Beams for eval generation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(data_dir: str, approach: str, split: str) -> Dataset:
    """Load one cleaned split from src.txt / tgt.txt."""
    base = Path(data_dir) / approach / "cleaned" / split / "cleaned"
    src_lines = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt_lines = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(tgt_lines), (
        f"{split}: src/tgt line count mismatch ({len(src_lines)} vs {len(tgt_lines)})"
    )
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def add_arn_latn_token(tokenizer, model):
    """
    Add arn_Latn as a new language token if absent, and initialize its embedding
    as the mean of ayr_Latn, grn_Latn, and quy_Latn.
    """
    if "arn_Latn" in tokenizer.get_vocab():
        logger.info("arn_Latn already in vocabulary — skipping initialization.")
        return

    logger.info("Adding arn_Latn to tokenizer and initializing embedding.")
    tokenizer.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    model.resize_token_embeddings(len(tokenizer))

    proxy_ids = [tokenizer.convert_tokens_to_ids(l) for l in PROXY_LANGS]
    new_id = tokenizer.convert_tokens_to_ids("arn_Latn")

    with torch.no_grad():
        shared = model.model.shared.weight
        avg_emb = shared[proxy_ids].mean(dim=0)
        shared[new_id] = avg_emb
        # lm_head shares weights in NLLB, but resize_token_embeddings ties them;
        # set explicitly if somehow not tied
        if hasattr(model, "lm_head") and model.lm_head.weight.data_ptr() != shared.data_ptr():
            model.lm_head.weight[new_id] = avg_emb

    logger.info(
        f"arn_Latn (id={new_id}) initialized from mean of "
        + ", ".join(f"{l}({i})" for l, i in zip(PROXY_LANGS, proxy_ids))
    )


def build_preprocess_fn(tokenizer, src_lang_code, tgt_lang_code, max_length):
    def preprocess(examples):
        tokenizer.src_lang = src_lang_code
        model_inputs = tokenizer(
            examples["src"],
            text_target=examples["tgt"],
            max_length=max_length,
            truncation=True,
        )
        return model_inputs
    return preprocess


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 (padding label) with pad token ID before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        return {
            "chrf": round(chrf.score, 2),
            "bleu": round(bleu.score, 2),
        }
    return compute_metrics


def main():
    args = parse_args()
    assert args.src != args.tgt, "Source and target languages must differ."

    src_code = LANG_CODES[args.src]
    tgt_code = LANG_CODES[args.tgt]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model     : {args.model}")
    logger.info(f"Direction : {args.src} ({src_code}) → {args.tgt} ({tgt_code})")
    logger.info(f"Approach  : {args.approach}")
    logger.info(f"Output    : {output_dir}")

    # Save run config for reproducibility
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Add arn_Latn if translating to/from Mapudungun
    add_arn_latn_token(tokenizer, model)

    # Set forced BOS token (target language) for generation
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    model.config.forced_bos_token_id = tgt_token_id
    logger.info(f"forced_bos_token_id = {tgt_token_id} ({tgt_code})")

    # Load and tokenize data
    logger.info("Loading data...")
    raw = {split: load_split(args.data_dir, args.approach, split)
           for split in ("train", "dev", "test")}
    logger.info(f"  train: {len(raw['train']):,}  dev: {len(raw['dev']):,}  test: {len(raw['test']):,}")

    preprocess = build_preprocess_fn(tokenizer, src_code, tgt_code, args.max_length)
    tokenized = {
        split: ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])
        for split, ds in raw.items()
    }

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model...")
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.predict(
        tokenized["test"],
        metric_key_prefix="test",
        num_beams=args.num_beams,
        max_length=args.max_length,
    )
    logger.info(f"Test results: {test_results.metrics}")
    (output_dir / "test_results.json").write_text(
        json.dumps(test_results.metrics, indent=2)
    )


if __name__ == "__main__":
    main()
