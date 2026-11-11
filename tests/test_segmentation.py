"""Segmentation must actually segment, and removing the @@ markers must give
back the original text. The Optuna test replays a search trial end to end:
segment_bpe.py at two vocabulary sizes, then the training loader."""

import itertools
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers
from helpers import REPO

STEMS = ["kutran", "mapu", "dungu", "ngütram", "küme", "lamngen", "ruka", "wentru", "domo", "kimün"]
SUFFIXES = ["", "ka", "kawün", "ngey", "lay", "fiñ", "keyu", "mew", "ngen", "tulay", "ün", "pe"]


def toy_corpus(n_lines=400, seed=0):
    rng = random.Random(seed)
    words = [s + x for s, x in itertools.product(STEMS, SUFFIXES)]
    return [" ".join(rng.choice(words) for _ in range(rng.randint(3, 9))) + "." for _ in range(n_lines)]


class TestMorfessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import morfessor
        import segment_morfessor as sm
        cls.sm = sm
        random.seed(0)
        counts = {}
        for line in toy_corpus():
            for w in sm.tokenize_words(line):
                counts[w.lower()] = counts.get(w.lower(), 0) + 1
        cls.model = morfessor.BaselineModel()
        cls.model.load_data([(c, w) for w, c in counts.items()], count_modifier=lambda x: x)
        cls.model.train_batch()

    def test_round_trip_and_segments(self):
        lines = toy_corpus(50, seed=1) + ["Kutrankawün Mapu ngütramkaleyu."]
        segmented = [self.sm.segment_line(line, self.model) for line in lines]
        for line, seg in zip(lines, segmented):
            self.assertEqual(self.sm.desegment(seg).rstrip("\n"), line)
        self.assertTrue(any("@@" in s for s in segmented), "Morfessor produced no boundaries")


class TestMorfessorVC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import segment_morfessor_vc as vc
        cls.vc = vc

    def test_parse_render_round_trip(self):
        line = "mapu@@ dun@@ gun kimi@@ el ruka\n"
        self.assertEqual(self.vc.parse_morfessor_line(line), [["mapu", "dun", "gun"], ["kimi", "el"], ["ruka"]])
        self.assertEqual(self.vc.render_line(self.vc.parse_morfessor_line(line)), line)

    def test_merges_only_in_vocab(self):
        vocab = {"▁mapudun", "gun"}
        self.assertEqual(self.vc.cascade_merge_word(["mapu", "dun", "gun"], vocab), ["mapudun", "gun"])
        self.assertEqual(self.vc.cascade_merge_word(["mapu", "dun"], set()), ["mapu", "dun"])
        self.assertEqual(self.vc.cascade_merge_word(["kimi", "el"], {"kimiel"}), ["kimiel"])

    def test_cascade_keeps_text_and_never_adds_boundaries(self):
        from segment_morfessor import desegment
        vocab = {"▁mapudun", "elmew", "▁ruka"}
        for line in ["mapu@@ dun@@ gun kimi@@ el@@ mew\n", "ruka@@ mew@@ ngen\n", "domo\n"]:
            out = self.vc.cascade_line(line, vocab)
            self.assertEqual(desegment(out), desegment(line))
            self.assertLessEqual(self.vc.boundary_count(out), self.vc.boundary_count(line))


class TestSentencePieceSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sentencepiece as spm
        import segment_bpe as sb
        cls.sb = sb
        cls.tmp = tempfile.TemporaryDirectory()
        corpus = Path(cls.tmp.name) / "train.txt"
        corpus.write_text("\n".join(toy_corpus()) + "\n", encoding="utf-8")
        cls.models = {}
        for size in (60, 140):
            prefix = Path(cls.tmp.name) / f"bpe_{size}"
            sb.train_sp_model([corpus], prefix, size)
            sp = spm.SentencePieceProcessor()
            sp.Load(str(prefix) + ".model")
            cls.models[size] = sp

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pieces_to_at_at(self):
        self.assertEqual(self.sb.pieces_to_at_at(["▁mapu", "dun", "gun", "▁ruka"]), "mapu@@ dun@@ gun ruka")
        self.assertEqual(self.sb.pieces_to_at_at([]), "")

    def test_round_trip(self):
        from segment_morfessor import desegment
        for size, sp in self.models.items():
            for line in toy_corpus(30, seed=2):
                with self.subTest(size=size, line=line):
                    self.assertEqual(desegment(self.sb.segment_line(sp, line)).rstrip("\n"), line)

    def test_vocab_size_changes_segmentation(self):
        lines = toy_corpus(30, seed=3)
        small = [self.sb.segment_line(self.models[60], l) for l in lines]
        large = [self.sb.segment_line(self.models[140], l) for l in lines]
        self.assertNotEqual(small, large)
        self.assertGreater(sum(s.count("@@") for s in small), sum(s.count("@@") for s in large))


class TestOptunaTrialPath(unittest.TestCase):
    """What a search trial did: segment_bpe.py --mode large --out-dir-name optuna_bpe,
    then finetune_nllb.load_split(..., "optuna_bpe"). The text that reaches training
    must contain markers and must differ between vocabulary sizes."""

    def test_trial_inputs_depend_on_vocab_size(self):
        import finetune_nllb as ft
        lines = toy_corpus()
        seen = {}
        for size in (60, 140):
            with tempfile.TemporaryDirectory() as root:
                for split in ("train", "dev", "test"):
                    d = Path(root) / "blocks" / "cleaned" / split / "cleaned"
                    d.mkdir(parents=True)
                    (d / "src.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    (d / "tgt.txt").write_text("\n".join(["es"] * len(lines)) + "\n", encoding="utf-8")
                subprocess.run([sys.executable, str(REPO / "scripts/tokenization/segment_bpe.py"),
                                "--mode", "large", "--large-vocab", str(size), "--out-dir-name", "optuna_bpe",
                                "--approaches", "blocks", "--data-root", root, "--bpe-root", str(Path(root) / "bpe")],
                               check=True, capture_output=True)
                seen[size] = ft.load_split(root, "blocks", "train", "arn", "optuna_bpe")["src"]
        for size, src in seen.items():
            self.assertTrue(any("@@" in s for s in src), f"vocab {size}: training input is unsegmented")
        self.assertNotEqual(seen[60], seen[140], "training input does not depend on vocab size")


if __name__ == "__main__":
    unittest.main()
