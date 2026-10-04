import unittest
from app.core.gt_loader import GTRange
from app.core.matcher import compute_chunk_expected_gt, classify_match
from app.core.types import VLMEvent

class TestGTMatcher(unittest.TestCase):
    def setUp(self):
        self.gt_ranges = [
            GTRange(start_ms=8000, end_ms=13000, score=0.90, description="No harness at edge"),
            GTRange(start_ms=41000, end_ms=46000, score=0.55, description="Forklift near pedestrian"),
        ]

    def test_overlap_expected_score(self):
        # Window [8000, 13000] overlaps first range exactly
        exp_score, desc = compute_chunk_expected_gt(8000, 13000, self.gt_ranges)
        self.assertEqual(exp_score, 0.90)
        self.assertEqual(desc, "No harness at edge")

        # Window [0, 5000] has no overlap -> expected 0.0
        exp_score, desc = compute_chunk_expected_gt(0, 5000, self.gt_ranges)
        self.assertEqual(exp_score, 0.0)

    def test_classify_match(self):
        # MATCH: abs(0.82 - 0.90) = 0.08 <= 0.25
        res = classify_match(0.82, 0.90, tol=0.25, high_thr=0.66)
        self.assertEqual(res, "MATCH")

        # MISS: expected=0.90 (>=0.66), vlm=0.10 (< 0.90 - 0.25 = 0.65)
        res = classify_match(0.10, 0.90, tol=0.25, high_thr=0.66)
        self.assertEqual(res, "MISS")

        # FALSE POSITIVE: expected=0.0 (< 0.66), vlm=0.75 (> 0.0 + 0.25 = 0.25)
        res = classify_match(0.75, 0.0, tol=0.25, high_thr=0.66)
        self.assertEqual(res, "FP")

        # SCORE OFF: both moderate/high but diff > tol
        res = classify_match(0.70, 0.40, tol=0.25, high_thr=0.66)
        # expected=0.4 (< 0.66), vlm=0.70 (> 0.65 -> FP)
        # Another case: expected=0.67, vlm=0.40 (< 0.67-0.25 = 0.42 -> MISS)
        # Test OFF: expected=0.5, vlm=0.2 (diff=0.3 > 0.25, exp < 0.66, vlm not > 0.75)
        res_off = classify_match(0.20, 0.50, tol=0.25, high_thr=0.66)
        self.assertEqual(res_off, "OFF")

if __name__ == "__main__":
    unittest.main()
