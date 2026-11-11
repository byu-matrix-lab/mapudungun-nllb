"""Markers: predictions are desegmented only when the output is Mapudungun, no
scored prediction contains @@, and COMET gets the raw (unsegmented) source."""

import json
import re
import unittest

import helpers
from helpers import HAVE_ARCHIVE, PREDICTIONS, REPO, V2_TEST, read_lines, table_cells

ARCHIVE_V2_PRED = helpers.ARCHIVE / "mapudungun_v2/predictions"


class TestPredictConfig(unittest.TestCase):
    def test_desegment_only_for_mapudungun_output(self):
        import predict_test as pt
        for key, cfg in pt.TEST_DATA.items():
            with self.subTest(key=key):
                expected = key.startswith("es-arn-")
                self.assertEqual(cfg.get("desegment_output", False), expected)

    def test_deseg_regex(self):
        import predict_test as pt
        self.assertEqual(pt.DESEG_RE.sub("", "kutran@@ ka@@ wün mew"), "kutrankawün mew")
        self.assertEqual(pt.DESEG_RE.sub("", "kutran@@"), "kutran")

    def test_language_code_stripped_from_output(self):
        import predict_test as pt
        self.assertEqual(pt.LANG_CODE_RE.sub("", "spa_Latn Vamos a conversar."), "Vamos a conversar.")
        self.assertEqual(pt.LANG_CODE_RE.sub("", "Vamos a conversar."), "Vamos a conversar.")


@unittest.skipUnless(HAVE_ARCHIVE, "archived predictions not mounted")
class TestScoredPredictions(unittest.TestCase):
    def test_no_markers_in_scored_predictions(self):
        # Test sentence 2236 ends in the corpus noise "@@@@@@@@", which zero-shot models copy
        # into es->arn output. A run of 4+ @ is that noise; any other @@ is a leaked marker
        # (seen as "kutran@@ ka", "word@@.", "cu@@pan", "¿@@ " in early runs).
        def leaked(line):
            return "@@" in re.sub(r"@{4,}", "", line)
        files = [PREDICTIONS / f"{c['file']}_preds.txt" for c in table_cells()]
        files += sorted(ARCHIVE_V2_PRED.glob("*_preds.txt"))
        for f in files:
            with self.subTest(file=f.name):
                leaks = [i for i, line in enumerate(read_lines(f)) if leaked(line)]
                self.assertEqual(leaks, [])
        # Negative control: an early run, not used by the table, that did leak markers.
        stale = read_lines(PREDICTIONS / "600M-blocks-arn-es-morfessor_bpe_preds.txt")
        self.assertGreater(sum(map(leaked, stale)), 1000)

    def test_segmented_sources_round_trip_to_raw(self):
        raw = read_lines(V2_TEST / "src.txt")
        self.assertFalse(any("@@" in line for line in raw), "raw test source contains markers")
        rescored = json.loads((REPO / "results/v2_comet_clean_src.json").read_text(encoding="utf-8"))
        segmented = [c["file"] for c in table_cells() if c["direction"] == "arn-es"
                     and any("@@" in r["src"] for r in json.loads(
                         (PREDICTIONS / f"{c['file']}_results.json").read_text(encoding="utf-8"))[:50])]
        self.assertEqual(sorted(segmented), sorted(rescored), "COMET rescoring must cover every segmented row")
        for stem in segmented:
            rows = json.loads((PREDICTIONS / f"{stem}_results.json").read_text(encoding="utf-8"))
            with self.subTest(file=stem):
                self.assertEqual([re.sub(r"@@ ?", "", r["src"]) for r in rows], raw)


if __name__ == "__main__":
    unittest.main()
