"""
Unit tests for HermesStreamingRunner._parse() — no GPU required.
"""
import unittest
import numpy as np

from app.core.types import VideoChunk, InvalidVLMReply
from vlm.hermes_runner import HermesStreamingRunner


class TestHermesRunnerParse(unittest.TestCase):
    def test_valid_json(self):
        result = HermesStreamingRunner._parse('{"caption": "Safe scene", "score": 0.1}')
        self.assertEqual(result.caption, "Safe scene")
        self.assertAlmostEqual(result.score, 0.1)

    def test_json_with_prefix_text(self):
        raw = 'Analysis complete: {"caption": "Worker near hazard", "score": 0.72}'
        result = HermesStreamingRunner._parse(raw)
        self.assertEqual(result.caption, "Worker near hazard")
        self.assertAlmostEqual(result.score, 0.72)

    def test_json_with_suffix_text(self):
        raw = '{"caption": "All clear", "score": 0.05} Please verify.'
        result = HermesStreamingRunner._parse(raw)
        self.assertEqual(result.caption, "All clear")
        self.assertAlmostEqual(result.score, 0.05)

    def test_no_json_raises_invalid(self):
        with self.assertRaises(InvalidVLMReply):
            HermesStreamingRunner._parse("No JSON here at all")

    def test_malformed_json_raises_invalid(self):
        with self.assertRaises(InvalidVLMReply):
            HermesStreamingRunner._parse('{"caption": broken json}}')

    def test_score_zero(self):
        result = HermesStreamingRunner._parse('{"caption": "Normal", "score": 0.0}')
        self.assertAlmostEqual(result.score, 0.0)

    def test_score_one(self):
        result = HermesStreamingRunner._parse('{"caption": "Critical", "score": 1.0}')
        self.assertAlmostEqual(result.score, 1.0)

    def test_missing_caption_uses_empty_string(self):
        result = HermesStreamingRunner._parse('{"score": 0.5}')
        self.assertEqual(result.caption, "")
        self.assertAlmostEqual(result.score, 0.5)


if __name__ == "__main__":
    unittest.main()
