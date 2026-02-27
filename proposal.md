# Bringing Mapudungun into the Modern MT Ecosystem 

**Proposed submission:** 8-page paper to AmericasNLP 2026 @ ACL on April 15, 2026  
**Proposed team:** Isaac Thompson & Dr. Brandon M. Rogers (BYU Spanish & Portuguese)  
**Last Updated:** Feb 25, 2026  
**Reviewed by:** Steve Richardson on Feb 23, 2026 and Eric Ringger on Feb 25, 2026 

---

## 1. Problem & Motivation 
Mapudungun, spoken by 200,000+ Mapuche people in Chile and Argentina, is absent from all major MT systems despite a 266,300-pair parallel corpus (Duan et al., LREC 2020). Its polysynthetic morphology produces a 3.3x word-type ratio over Spanish, suggesting that standard BPE tokenization is poorly suited to the language. No modern multilingual model has been fine-tuned on it, and the speech data in the corpus is out of scope for this submission but represents clear future work.

## 2. Prior Work & Gap 
* **Duan et al. (LREC 2020):** 12.9 BLEU (es→arn) using a small joint BPE vocabulary (5,000 shared subwords); never revisited with modern models.
* **Ahumada et al. (BEA 2022):** Orthography converter and morphological segmenter, but no MT system.
* **Pendas et al. (AmericasNLP 2023):** Fine-tuned MarianMT (pivoting from a Spanish-German checkpoint) on ~30K pairs using active learning strategies, achieving 65.45 BLEU (es→arn) on 60% of their data. Results are not directly comparable to Duan et al. due to different dataset sizes and evaluation setups. No modern pretrained multilingual model was used.

To our knowledge, no published work combines modern pretrained multilingual models with morphology-aware tokenization for Mapudungun.

## 3. Proposed Approach 
We fine-tune NLLB-200-distilled-600M bidirectionally on the full 266K corpus. The central contribution is a tokenization study comparing:
* **(a)** Duan et al.'s original 5K joint BPE
* **(b)** A larger language-specific BPE vocabulary
* **(c)** Morfessor segmentation (an unsupervised morpheme segmentation tool well-suited to polysynthetic languages)
* **(d)** A linguistically-informed segmentation approach developed in collaboration with Dr. Rogers, grounding the NLP methodology in fieldwork expertise.

An LLM prompting baseline (GPT-4o, Claude; zero-shot through grammar-augmented) tests whether in-context learning can substitute for fine-tuning on a language absent from LLM training data. Novelty is threefold: first application of a massively multilingual model to Mapudungun, a systematic tokenization study for a polysynthetic language with real linguistic grounding, and an LLM benchmark establishing a new baseline for the field.

## 4. Experimental Plan 
* **Data/metrics:** 240K/13K/13K train/dev/test splits; primary metric chrF++, secondary BLEU and tokenizer fertility.
* **Baselines:** Duan et al. 2020 Transformer, zero-shot NLLB-200, LLM prompting variants.
* **Linguistic analysis (Rogers):** Error analysis across morphological categories, code-switching behavior, and dialect variation (Nguluche, Lafkenche, Pewenche), drawing on his published work on Spanish-Mapudungun contact and phonetics.

## 5. Validation & Impact 
* **Predictions / Definition of Success:** NLLB fine-tuning meaningfully outperforms the 2020 baseline; linguistically-informed tokenization outperforms Duan et al.'s 5K joint BPE; LLM prompting lags behind fine-tuning.
* **Deliverables:** First NLLB-compatible Mapudungun MT model (HuggingFace) and tokenization guidance transferable to other polysynthetic AmericasNLP languages.
* **Future Work:** The 142-hour speech corpus positions this work for future ASR and speech translation research.