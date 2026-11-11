#!/usr/bin/env python3
"""Translate with a released model and its condition-specific preprocessing."""

import argparse
import json
import re
from pathlib import Path

import morfessor
import sentencepiece as spm
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_morfessor_line(line):
    words, current = [], []
    for token in line.rstrip("\n").split():
        current.append(token[:-2] if token.endswith("@@") else token)
        if not token.endswith("@@"):
            words.append(current)
            current = []
    if current:
        words.append(current)
    return words


def render_word(parts):
    return " ".join(part + ("@@" if i < len(parts) - 1 else "") for i, part in enumerate(parts))


def morfessor_segment(text, model):
    rendered = []
    for word in text.split():
        parts, _ = model.viterbi_segment(word.lower())
        if len(parts) == 1:
            rendered.append(word)
            continue
        offsets, pos = [], 0
        for part in parts:
            offsets.append(word[pos:pos + len(part)])
            pos += len(part)
        rendered.append(render_word(offsets))
    return " ".join(rendered)


def sp_segment(text, processor):
    output = []
    for piece in processor.EncodeAsPieces(text):
        if piece.startswith("▁"):
            output.append(piece[1:])
        else:
            if output:
                output[-1] += "@@"
            output.append(piece)
    return " ".join(output)


def morfessor_bpe_segment(text, morph, processor):
    words = parse_morfessor_line(morfessor_segment(text, morph))
    rendered = []
    for word in words:
        pieces = []
        for morpheme in word:
            encoded = processor.EncodeAsPieces(morpheme)
            pieces.extend([piece.lstrip("▁") for piece in encoded if piece.lstrip("▁")])
        rendered.append(render_word(pieces))
    return " ".join(rendered)


def vc_segment(text, morph, processor):
    vocab = {processor.IdToPiece(i) for i in range(processor.GetPieceSize())}
    rendered = []
    for parts in parse_morfessor_line(morfessor_segment(text, morph)):
        merged, i = list(parts), 0
        while i < len(merged) - 1:
            candidate = merged[i] + merged[i + 1]
            if candidate in vocab or "▁" + candidate in vocab:
                merged[i:i + 2] = [candidate]
            else:
                i += 1
        rendered.append(render_word(merged))
    return " ".join(rendered)


def local_snapshot(model_id):
    path = Path(model_id).expanduser()
    return path if path.is_dir() else Path(snapshot_download(model_id))


def preprocess(text, condition, snapshot):
    if condition == "standard-bpe":
        return text
    morph_path = snapshot / "mapudungun_morfessor_model.bin"
    morph = morfessor.MorfessorIO().read_binary_model_file(str(morph_path)) if morph_path.exists() else None
    if condition == "morfessor":
        return morfessor_segment(text, morph)
    if condition == "morfessor-vc":
        processor = spm.SentencePieceProcessor(model_file=str(snapshot / "sentencepiece.bpe.model"))
        return vc_segment(text, morph, processor)
    segmenter = spm.SentencePieceProcessor(model_file=str(snapshot / "mapudungun_segmenter.model"))
    if condition == "morfessor-bpe":
        return morfessor_bpe_segment(text, morph, segmenter)
    return sp_segment(text, segmenter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("text", nargs="*", help="Text to translate; reads nonempty stdin lines if omitted")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    snapshot = local_snapshot(args.model)
    metadata = json.loads((snapshot / "release_metadata.json").read_text(encoding="utf-8"))
    texts = args.text
    if not texts:
        import sys
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if not texts:
        raise SystemExit("provide text as arguments or on stdin")

    direction, condition = metadata["direction"], metadata["condition"]
    prepared = [preprocess(text, condition, snapshot) if direction == "arn-es" else text for text in texts]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, src_lang=metadata["source_lang"])
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = tokenizer(prepared, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        generated = model.generate(**batch, forced_bos_token_id=tokenizer.convert_tokens_to_ids(metadata["target_lang"]),
                                   max_new_tokens=args.max_new_tokens)
    output = tokenizer.batch_decode(generated, skip_special_tokens=True)
    if direction == "es-arn" and condition != "standard-bpe":
        output = [re.sub(r"@@ ?", "", text) for text in output]
    print("\n".join(output))


if __name__ == "__main__":
    main()
