"""Bug 1 (published version): the arn->es Standard BPE baselines were trained
under transformers 5.2.0, which loaded NLLB as a generic tokenizer that ignored
src_lang, so no training example carried a language tag."""

import unittest

import helpers
from helpers import HAVE_NLLB


class TestTransformersVersion(unittest.TestCase):
    def test_pinned_major_version(self):
        import transformers
        self.assertTrue(transformers.__version__.startswith("4."),
                        f"transformers {transformers.__version__}: NLLB tagging breaks under 5.x")


@unittest.skipUnless(HAVE_NLLB, "NLLB-200 600M tokenizer not in the HF cache")
class TestLanguageTags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from datasets import Dataset
        from transformers import AutoTokenizer
        import finetune_nllb as ft
        cls.ft = ft
        cls.tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        cls.tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
        cls.pairs = Dataset.from_dict({
            "src": ["Pichi ngütramkayayu.", "Marimari lamngen."],
            "tgt": ["Vamos a conversar un poco.", "Buenos días hermana."],
        })

    def tokenize(self, src_code, tgt_code, ds):
        pre = self.ft.build_preprocess_fn(self.tok, src_code, tgt_code, 128)
        return {"train": ds.map(pre, batched=True, remove_columns=["src", "tgt"])}

    def test_tokenizer_class(self):
        self.assertIn("Nllb", type(self.tok).__name__)

    def test_arn_es_tags(self):
        tokenized = self.tokenize("arn_Latn", "spa_Latn", self.pairs)
        self.ft.check_language_tags(tokenized, self.tok, "arn_Latn", "spa_Latn")
        first = tokenized["train"][0]
        self.assertEqual(self.tok.convert_ids_to_tokens(first["input_ids"][0]), "arn_Latn")
        self.assertEqual(self.tok.convert_ids_to_tokens(first["labels"][0]), "spa_Latn")
        self.assertEqual(self.tok.convert_ids_to_tokens(first["input_ids"][-1]), "</s>")

    def test_es_arn_tags(self):
        swapped = self.pairs.rename_columns({"src": "tgt", "tgt": "src"})
        tokenized = self.tokenize("spa_Latn", "arn_Latn", swapped)
        self.ft.check_language_tags(tokenized, self.tok, "spa_Latn", "arn_Latn")

    def test_guard_rejects_untagged(self):
        tokenized = self.tokenize("arn_Latn", "spa_Latn", self.pairs)
        stripped = {"train": tokenized["train"].map(
            lambda e: {"input_ids": e["input_ids"][1:], "labels": e["labels"][1:]})}
        with self.assertRaises(AssertionError):
            self.ft.check_language_tags(stripped, self.tok, "arn_Latn", "spa_Latn")


if __name__ == "__main__":
    unittest.main()
