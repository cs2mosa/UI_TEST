"""
Integration and statefulness tests for Qwen3VLRunner on real hardware.
Verifies frame accounting, non-mutation during generation, and session reset.
"""

import os
import sys
import unittest
import cv2
import numpy as np

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
vlm_path = os.path.join(project_root, "vlm_test_ui")
if vlm_path not in sys.path:
    sys.path.insert(0, vlm_path)

from app.core.types import VideoChunk, VLMResult
from vlm.qwen3vl_runner import Qwen3VLRunner

def load_frames(video_path: str, max_frames: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR; convert to RGB numpy array
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


class TestQwen3VLRunnerState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[*] Initializing Qwen3VLRunner for statefulness testing...")
        cls.runner = Qwen3VLRunner(
            num_frames=2,
            max_new_tokens=60,
            load_in_4bit=True,
            memory_budget_frames=8,
            compress_to_frames=4,
            use_infinipot=True,
        )
        video_path = os.path.join(project_root, "vlm_test_ui", "examples", "synthetic_test.mp4")
        all_frames = load_frames(video_path, max_frames=4)
        assert len(all_frames) >= 4, f"Need at least 4 frames in test video, got {len(all_frames)}"
        
        cls.chunk1 = VideoChunk(
            chunk_id=0,
            source=video_path,
            start_ms=0,
            end_ms=1000,
            fps=2.0,
            frames=all_frames[:2],
            timestamps_ms=[0, 500],
        )
        cls.chunk2 = VideoChunk(
            chunk_id=1,
            source=video_path,
            start_ms=1000,
            end_ms=2000,
            fps=2.0,
            frames=all_frames[2:4],
            timestamps_ms=[1000, 1500],
        )

    def setUp(self):
        self.runner.reset()

    def test_01_sequential_chunks_and_frames_seen(self):
        """
        After two sequential __call__s without reset(), frames_seen must equal
        the sum of frames across both chunks, and captions must be non-empty/valid.
        """
        print("\n--- Running test_01_sequential_chunks_and_frames_seen ---")
        res1 = self.runner(self.chunk1)
        print(f"[+] Chunk 1 output: caption='{res1.caption}', score={res1.score}")
        self.assertIsInstance(res1, VLMResult)
        self.assertTrue(len(res1.caption) > 0, "Chunk 1 returned empty caption")
        self.assertEqual(self.runner.frames_seen, len(self.chunk1.frames))
        seq_len_after_chunk1 = self.runner.past_key_values.layers[0].keys.shape[2]

        res2 = self.runner(self.chunk2)
        print(f"[+] Chunk 2 output: caption='{res2.caption}', score={res2.score}")
        self.assertIsInstance(res2, VLMResult)
        self.assertTrue(len(res2.caption) > 0, "Chunk 2 returned empty caption")
        self.assertEqual(
            self.runner.frames_seen,
            len(self.chunk1.frames) + len(self.chunk2.frames),
            "frames_seen must equal sum of frames across both chunks"
        )
        seq_len_after_chunk2 = self.runner.past_key_values.layers[0].keys.shape[2]
        self.assertGreater(seq_len_after_chunk2, seq_len_after_chunk1, "KV cache must grow after second chunk")

    def test_02_non_mutation_during_generation(self):
        """
        Calling _generate_from_cache twice in a row must not grow self.past_key_values.
        """
        print("\n--- Running test_02_non_mutation_during_generation ---")
        self.runner._append_chunk_to_cache(self.chunk1)
        base_cache_len = self.runner.past_key_values.layers[0].keys.shape[2]

        # Call generate twice
        out1 = self.runner._generate_from_cache(self.runner.prompt)
        len_after_gen1 = self.runner.past_key_values.layers[0].keys.shape[2]
        self.assertEqual(base_cache_len, len_after_gen1, "First generation mutated persistent cache length!")

        out2 = self.runner._generate_from_cache(self.runner.prompt)
        len_after_gen2 = self.runner.past_key_values.layers[0].keys.shape[2]
        self.assertEqual(base_cache_len, len_after_gen2, "Second generation mutated persistent cache length!")
        self.assertTrue(len(out1) > 0)
        self.assertTrue(len(out2) > 0)

    def test_03_reset_clears_state_completely(self):
        """
        reset() must set past_key_values to None, frames_seen to 0, and a fresh call
        must behave identically to a first-ever call.
        """
        print("\n--- Running test_03_reset_clears_state_completely ---")
        self.runner(self.chunk1)
        self.assertIsNotNone(self.runner.past_key_values)
        self.assertGreater(self.runner.frames_seen, 0)

        # Call reset twice to verify idempotence
        self.runner.reset()
        self.runner.reset()

        self.assertIsNone(self.runner.past_key_values, "past_key_values must be None after reset")
        self.assertEqual(self.runner.frames_seen, 0, "frames_seen must be 0 after reset")
        self.assertEqual(self.runner.system_size, 0, "system_size must be 0 after reset")
        self.assertIsNone(self.runner.token_per_frame, "token_per_frame must be None after reset")

        # Fresh call after reset
        fresh_res = self.runner(self.chunk1)
        self.assertEqual(self.runner.frames_seen, len(self.chunk1.frames))
        self.assertIsNotNone(self.runner.past_key_values)
        self.assertTrue(len(fresh_res.caption) > 0)


if __name__ == "__main__":
    unittest.main()
