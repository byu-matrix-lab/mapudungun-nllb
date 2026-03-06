#!/usr/bin/env python3
"""
Generate a few hand-picked demo translations from the best fine-tuned model
in both directions for meeting/paper examples.
"""

import re
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

ARN_ES_MODEL = "/home/it238/nobackup/autodelete/mapudungun/models/nllb-200-3.3B-blocks-arn-es/best"
ES_ARN_MODEL = "/home/it238/nobackup/autodelete/mapudungun/models/nllb-200-3.3B-blocks-es-arn/best"
# Base NLLB-3.3B snapshot — needed for sentencepiece vocab file.
# arn_Latn is added manually below to match what finetune_nllb.py did.
BASE_TOKENIZER = "/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
ARN_CODE = "arn_Latn"
ES_CODE  = "spa_Latn"

ARN_ES_EXAMPLES = [
    ("Wingka üymu ta ni kiñe, kiñe lingue piam müli pu mallin tati.",
     "Tiene un nombre huinca; un lingue dicen en los pajonales."),
    ("Ka müleki ta kutran feyti enfermedad infecciosa ka müleki ka fey.",
     "También existe esa enfermedad infecciosa, también existe."),
    ("Iñche fey iñche ñi üy pingi Maria Isabel Coñuepan pingi.",
     "Mi nombre es Maria Isabel Coñuepan."),
    ("Ahora estoy controlando en la posta müten kontrolaletulan ta.",
     "Ahora estoy controlándome en la Posta no más."),
    ("Petu konünmun ta ngütrammu fey tukulpayayu kutran chumngechi chum chumleken kutran.",
     "Pero antes de entrar a nuestra conversación vamos a empezar por cómo están las enfermedades."),
]

ES_ARN_EXAMPLES = [
    ("¿Qué remedio es bueno, el remedio del hospital o los remedios naturales?",
     "Chem l'awen' ñi kümen. pital l'awen' kam mapu l'awen'."),
    ("Yo soy de Pitraco, en Pitraco está mi tierra.",
     "Inche kupan pitrako en pitrako mapu."),
    ("Mi nombre es Maria Isabel Coñuepan.",
     "Iñche fey iñche ñi üy pingi Maria Isabel Coñuepan pingi."),
    ("También existe esa enfermedad infecciosa.",
     "Ka müleki ta kutran feyti enfermedad infecciosa ka müleki ka fey."),
    ("Ahora me estoy controlando en la Posta no más.",
     "Ahora estoy controlando en la posta müten kontrolaletulan ta."),
]


def translate(model, tokenizer, texts, src_lang, tgt_lang, device):
    tokenizer.src_lang = src_lang
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=256)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    # Strip any residual lang-code prefix (e.g. "spa_Latn ") that skip_special_tokens misses
    lang_codes = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3,}\s*")
    return [lang_codes.sub("", s).strip() for s in decoded]


def run_direction(model_path, examples, src_lang, tgt_lang, label, device):
    print(f"\n{'='*70}")
    print(f"  {label}  (NLLB-3.3B fine-tuned)")
    print(f"{'='*70}")
    # Load base tokenizer (has sentencepiece vocab), then add arn_Latn to match
    # what finetune_nllb.py did during training (id=256204).
    tok = NllbTokenizer.from_pretrained(BASE_TOKENIZER)
    tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    srcs = [s for s, _ in examples]
    refs = [r for _, r in examples]
    preds = translate(mdl, tok, srcs, src_lang, tgt_lang, device)
    for src, ref, pred in zip(srcs, refs, preds):
        print(f"\n  Source:      {src}")
        print(f"  Reference:   {ref}")
        print(f"  Prediction:  {pred}")
    del mdl, tok
    torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    run_direction(ARN_ES_MODEL, ARN_ES_EXAMPLES, ARN_CODE, ES_CODE,
                  "Mapudungun → Spanish  (arn→es)", device)
    run_direction(ES_ARN_MODEL, ES_ARN_EXAMPLES, ES_CODE, ARN_CODE,
                  "Spanish → Mapudungun  (es→arn)", device)


if __name__ == "__main__":
    main()
