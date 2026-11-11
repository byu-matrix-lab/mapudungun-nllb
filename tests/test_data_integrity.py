"""The recovered original data and the v2 data must match their checksums, and
the recovered test split must equal the test set reconstructed from the
published predictions."""

import hashlib
import json
import unittest

import helpers
from helpers import ARCHIVE, REBUILD, V2_TEST

V2_DATA = ARCHIVE / "mapudungun_v2/data"


def check_sums(sums_file, base):
    """Entries are absolute paths or paths relative to `base`."""
    bad = []
    for line in sums_file.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = base / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


@unittest.skipUnless((REBUILD / "SHA256SUMS").exists(), "recovered data not mounted")
class TestRecoveredData(unittest.TestCase):
    def test_checksums(self):
        self.assertEqual(check_sums(REBUILD / "SHA256SUMS", REBUILD / "data-processed"), [])

    def test_test_split_is_the_published_one(self):
        rebuilt = REBUILD / "data-processed/blocks/cleaned/test/cleaned"
        for name in ("src.txt", "tgt.txt"):
            with self.subTest(file=name):
                self.assertEqual((rebuilt / name).read_bytes(), (V2_TEST / name).read_bytes())

    def test_split_sizes(self):
        base = REBUILD / "data-processed/blocks/cleaned"
        stats = json.loads((helpers.REPO / "results/paper_stats.json").read_text())["paper_v2"]["data"]
        for split in ("train", "dev", "test"):
            n = stats[f"{split}|arn"]["sents"]
            for side in ("src.txt", "tgt.txt"):
                with self.subTest(split=split, side=side):
                    self.assertEqual(len(helpers.read_lines(base / split / "cleaned" / side)), n)


@unittest.skipUnless((V2_DATA / "SHA256SUMS").exists(), "v2 data not mounted")
class TestV2Data(unittest.TestCase):
    def test_checksums(self):
        self.assertEqual(check_sums(V2_DATA / "SHA256SUMS", V2_DATA), [])


if __name__ == "__main__":
    unittest.main()
