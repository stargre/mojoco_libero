import json
import unittest

from libero_saycan.qwen import QwenRelevanceScorer


class QwenParsingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "Pick:0:bowl_1", "name": "Pick", "args": {"object": "bowl_1"}},
            {"id": "Done", "name": "Done", "args": {}},
        ]

    def test_parses_all_candidate_ids(self):
        payload = json.dumps({
            "Pick:0:bowl_1": {"relevance": 0.96, "rationale": "requested object"},
            "Done": {"relevance": 0.02, "rationale": "goal not met"},
        })
        result = QwenRelevanceScorer.parse_response(payload, self.candidates)
        self.assertAlmostEqual(result["Pick:0:bowl_1"]["relevance"], 0.96)
        self.assertEqual(set(result.keys()), {"Pick:0:bowl_1", "Done"})

    def test_rejects_missing_candidate(self):
        payload = json.dumps({
            "Pick:0:bowl_1": {"relevance": 0.9},
        })
        with self.assertRaisesRegex(ValueError, "candidate IDs mismatch"):
            QwenRelevanceScorer.parse_response(payload, self.candidates)

    def test_rejects_invalid_relevance(self):
        payload = json.dumps({
            "Pick:0:bowl_1": {"relevance": 1.2},
            "Done": {"relevance": 0.0},
        })
        with self.assertRaisesRegex(ValueError, "must be finite"):
            QwenRelevanceScorer.parse_response(payload, self.candidates)


if __name__ == "__main__":
    unittest.main()
