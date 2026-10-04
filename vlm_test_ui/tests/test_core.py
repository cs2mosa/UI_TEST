import unittest
import numpy as np
from app.core.chunker import compute_chunk_windows
from app.core.types import VideoChunk, VLMResult, InvalidVLMReply
from app.core.vlm_adapter import VLMAdapter

class TestChunker(unittest.TestCase):
    def test_empty_or_zero(self):
        self.assertEqual(compute_chunk_windows(0), [])
        self.assertEqual(compute_chunk_windows(-100), [])

    def test_short_video(self):
        # 3000ms is shorter than 5000ms chunk
        windows = compute_chunk_windows(3000, chunk_s=5.0, overlap_s=1.0)
        self.assertEqual(windows, [(0, 3000)])

    def test_multi_chunk_deterministic(self):
        # 13000ms with 5s chunk and 1s overlap (stride = 4s = 4000ms)
        # Windows: [0, 5000], [4000, 9000], [8000, 13000]
        windows = compute_chunk_windows(13000, chunk_s=5.0, overlap_s=1.0)
        self.assertEqual(windows, [
            (0, 5000),
            (4000, 9000),
            (8000, 13000)
        ])

    def test_partial_last_window(self):
        # 10500ms with 5s chunk, 1s overlap -> stride 4s
        # [0, 5000], [4000, 9000], [8000, 10500]
        windows = compute_chunk_windows(10500, chunk_s=5.0, overlap_s=1.0, include_partial=True)
        self.assertEqual(windows, [
            (0, 5000),
            (4000, 9000),
            (8000, 10500)
        ])

class TestVLMAdapter(unittest.TestCase):
    def setUp(self):
        self.dummy_chunk = VideoChunk(
            chunk_id=0,
            source="test.mp4",
            start_ms=0,
            end_ms=5000,
            fps=4.0,
            frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            timestamps_ms=[0]
        )

    def test_vlm_result(self):
        adapter = VLMAdapter(lambda c: VLMResult("Safe scene", 0.1))
        event = adapter.process_chunk(self.dummy_chunk, 1)
        self.assertEqual(event.status, "ok")
        self.assertEqual(event.caption, "Safe scene")
        self.assertEqual(event.score, 0.1)
        adapter.shutdown()

    def test_vlm_tuple(self):
        adapter = VLMAdapter(lambda c: ("Violation", 0.85))
        event = adapter.process_chunk(self.dummy_chunk, 1)
        self.assertEqual(event.status, "ok")
        self.assertEqual(event.score, 0.85)
        adapter.shutdown()

    def test_out_of_bounds_score_flagged(self):
        # Design Doc §6: out-of-range values are flagged, not clamped
        adapter = VLMAdapter(lambda c: ("Extreme", 1.5))
        event = adapter.process_chunk(self.dummy_chunk, 1)
        self.assertEqual(event.status, "invalid")
        self.assertIn("out of expected range", event.error)
        adapter.shutdown()

    def test_exception_handling(self):
        def bad_vlm(c):
            raise RuntimeError("Model crash")
        adapter = VLMAdapter(bad_vlm)
        event = adapter.process_chunk(self.dummy_chunk, 1)
        self.assertEqual(event.status, "error")
        self.assertIn("Model crash", event.error)
        adapter.shutdown()

    def test_latency_per_frame(self):
        chunk = VideoChunk(
            chunk_id=0, source="test.mp4", start_ms=0, end_ms=5000, fps=4.0,
            frames=[np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)],
            timestamps_ms=[0, 1000, 2000, 3000]
        )
        adapter = VLMAdapter(lambda c: VLMResult("Safe scene", 0.1))
        event = adapter.process_chunk(chunk, 1)
        self.assertEqual(event.num_frames, 4)
        self.assertIsNotNone(event.latency_per_frame_ms)
        self.assertGreater(event.latency_per_frame_ms, 0)
        adapter.shutdown()

if __name__ == "__main__":
    unittest.main()
