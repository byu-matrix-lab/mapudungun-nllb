"""Scoring: the v2 scripts compute chrF++ from sacrebleu's sufficient statistics
(so the bootstrap can resample sentences). That must equal sacrebleu exactly,
for both the corpus score and each sentence score. The published tables are
the mean of the sentence scores; they must be reproducible from the files."""

import json
import unittest

import numpy as np
from sacrebleu.metrics import CHRF

import helpers
from helpers import HAVE_ARCHIVE, PREDICTIONS, table_cells

PREDS = ["Vamos a conversar un poco.", "Así es, hermana.", "No sé qué decir", "", "Sí, sí, sí."]
REFS = ["Vamos a conversar un poco.", "Así es pues hermana.", "No sé qué decirte.", "Bueno.", "Sí."]


class TestStatsPath(unittest.TestCase):
    def setUp(self):
        self.chrf = CHRF(word_order=2)

    def score(self, stats):
        return self.chrf._compute_score_from_stats(list(stats)).score

    def test_corpus_from_stats_equals_sacrebleu(self):
        st = np.array(self.chrf._extract_corpus_statistics(PREDS, [REFS]))
        self.assertAlmostEqual(self.score(st.sum(axis=0)), self.chrf.corpus_score(PREDS, [REFS]).score, places=9)

    def test_sentence_from_stats_equals_sacrebleu(self):
        st = np.array(self.chrf._extract_corpus_statistics(PREDS, [REFS]))
        for row, p, r in zip(st, PREDS, REFS):
            self.assertAlmostEqual(self.score(row), self.chrf.sentence_score(p, [r]).score, places=9)

    def test_metric_is_chrf_plus_plus(self):
        self.assertEqual(self.chrf.word_order, 2)
        self.assertNotAlmostEqual(CHRF().corpus_score(PREDS, [REFS]).score,
                                  self.chrf.corpus_score(PREDS, [REFS]).score)


@unittest.skipUnless(HAVE_ARCHIVE, "archived predictions not mounted")
class TestPublishedTable(unittest.TestCase):
    def test_every_cell_reproduces_as_sentence_mean(self):
        cells = table_cells()
        self.assertEqual(len(cells), 58)
        chrf = CHRF(word_order=2)
        for c in cells[::7]:  # a spread of cells; v2_phaseA_audit.py checks all 58
            rows = json.loads((PREDICTIONS / f"{c['file']}_results.json").read_text(encoding="utf-8"))
            mean = np.mean([chrf.sentence_score(r["pred"], [r["ref"]]).score for r in rows])
            with self.subTest(cell=c["file"]):
                self.assertAlmostEqual(round(mean, 2), c["table"], delta=0.006)


if __name__ == "__main__":
    unittest.main()
