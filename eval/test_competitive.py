"""Stdlib-only acceptance tests for the submitted prosecutor."""
from __future__ import annotations

import unittest

from eval.prosecute import load_fixtures, prosecute, score_prosecutor


class CompetitiveProsecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        cls.report = score_prosecutor(prosecute, cls.fixtures)

    def test_competitive_thresholds(self):
        report = self.report
        self.assertEqual(report["n_errors"], 0)
        self.assertEqual(report["n_timeouts"], 0)
        self.assertEqual(report["rejected"], 0)
        self.assertGreaterEqual(report["precision"], 0.85)
        self.assertGreaterEqual(report["recall"], 0.65)
        self.assertGreaterEqual(report["f1"], 0.72)

    def test_all_clean_fixtures_are_silent(self):
        for fixture in self.fixtures:
            if not fixture.get("label", {}).get("present_classes"):
                result = prosecute(fixture["trace"], fixture["answer"], fixture["card"])
                self.assertEqual(result["claims"], [], fixture["fixture_id"])

    def test_every_near_miss_uses_complete_proof(self):
        for fixture in self.fixtures:
            if fixture.get("variant") != "near_miss":
                continue
            result = prosecute(fixture["trace"], fixture["answer"], fixture["card"])
            by_class = {claim["cls"]: claim for claim in result["claims"]}
            for cls, truth in fixture["label"]["present_classes"].items():
                self.assertIn(cls, by_class, fixture["fixture_id"])
                self.assertTrue(set(truth["proof_refs"]) <= set(by_class[cls]["evidence"]))


if __name__ == "__main__":
    unittest.main()
