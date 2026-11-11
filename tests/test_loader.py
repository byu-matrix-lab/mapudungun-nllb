"""Bug 2: load_split silently fell back to the unsegmented data for any
tokenizer approach it did not list, so every Optuna trial trained on
unsegmented text."""

import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  (sets sys.path)
import finetune_nllb as ft


def write_pair(base, tag):
    base.mkdir(parents=True, exist_ok=True)
    (base / "src.txt").write_text(f"arn {tag}\n", encoding="utf-8")
    (base / "tgt.txt").write_text(f"es {tag}\n", encoding="utf-8")


class TestLoadSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "blocks"
        write_pair(root / "cleaned" / "train" / "cleaned", "standard")
        write_pair(root / "morfessor" / "train", "morfessor")
        for name in ft.SEGMENTED_APPROACHES:
            write_pair(root / name / "train", name)
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, approach, src="arn"):
        ds = ft.load_split(self.root, "blocks", "train", src, approach)
        return ds[0]["src"], ds[0]["tgt"]

    def test_each_approach_reads_its_own_directory(self):
        for name in ("standard", "morfessor", *ft.SEGMENTED_APPROACHES):
            with self.subTest(name=name):
                self.assertEqual(self.load(name), (f"arn {name}", f"es {name}"))

    def test_es_arn_swaps_sides(self):
        self.assertEqual(self.load("morfessor_vc", src="es"), ("es morfessor_vc", "arn morfessor_vc"))

    def test_unknown_approach_raises(self):
        for name in ("optuna", "cascade", "duan_bpe", "large_bpe", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load(name)

    def test_missing_segmented_data_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                ft.load_split(empty, "blocks", "train", "arn", "optuna_bpe")


if __name__ == "__main__":
    unittest.main()
