"""
Unit tests for timeout and threading safety in VLMAdapter (Phase 4).
Verifies:
1. Timeout is session-fatal for stateful VLMs (calls reset() and sets session_reset=True).
2. Stateless callables without reset() do not set session_reset.
3. Real runner recovery: after a timeout triggers reset(), subsequent chunk processes cleanly.
"""

import os
import sys
import time
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
vlm_path = os.path.join(project_root, "vlm_test_ui")
if vlm_path not in sys.path:
    sys.path.insert(0, vlm_path)

from app.core.types import VideoChunk, VLMResult, VLMEvent
from app.core.vlm_adapter import VLMAdapter


class StatefulTimeoutStub:
    """Simulates a slow VLM that sleeps past timeout and tracks reset() calls."""
    def __init__(self, delay_s: float = 0.5):
        self.delay_s = delay_s
        self.reset_count = 0
        self.call_count = 0

    def reset(self):
        self.reset_count += 1

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        self.call_count += 1
        time.sleep(self.delay_s)
        return VLMResult(caption="Slow result", score=0.0)


class StatelessTimeoutStub:
    """Simulates a slow VLM without a reset method."""
    def __init__(self, delay_s: float = 0.5):
        self.delay_s = delay_s

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        time.sleep(self.delay_s)
        return VLMResult(caption="Stateless slow", score=0.0)


class TestVLMTimeoutSafety(unittest.TestCase):
    def setUp(self):
        dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.dummy_chunk = VideoChunk(
            chunk_id=0,
            source="synthetic.mp4",
            start_ms=0,
            end_ms=1000,
            fps=1.0,
            frames=[dummy_frame],
            timestamps_ms=[0],
        )

    def test_simulated_timeout_calls_reset_and_sets_flag(self):
        """Simulated timeout on stateful VLM must invoke reset() and set event.session_reset = True."""
        stub = StatefulTimeoutStub(delay_s=0.3)
        adapter = VLMAdapter(stub, timeout_s=0.05)

        event = adapter.process_chunk(self.dummy_chunk, event_id=1)

        self.assertEqual(event.status, "timeout")
        self.assertTrue(event.session_reset, "session_reset must be True when stateful VLM times out")
        self.assertEqual(stub.reset_count, 1, "reset() must be called exactly once upon timeout")
        adapter.shutdown()

    def test_stateless_timeout_does_not_set_flag(self):
        """Stateless VLM timeout leaves session_reset = False."""
        stub = StatelessTimeoutStub(delay_s=0.3)
        adapter = VLMAdapter(stub, timeout_s=0.05)

        event = adapter.process_chunk(self.dummy_chunk, event_id=1)

        self.assertEqual(event.status, "timeout")
        self.assertFalse(event.session_reset, "session_reset must remain False for stateless VLM")
        adapter.shutdown()

    def test_real_runner_timeout_and_clean_recovery(self):
        """
        Trigger a real timeout against Qwen3VLRunner, verify session_reset is flagged,
        and confirm next chunk's __call__ recovers cleanly into a valid empty-cache state.
        """
        from vlm.qwen3vl_runner import Qwen3VLRunner
        runner = Qwen3VLRunner(
            num_frames=2,
            max_new_tokens=40,
            load_in_4bit=True,
            use_infinipot=True,
        )

        # Step 1: Force a timeout by setting timeout_s very small (0.001s)
        adapter = VLMAdapter(runner, timeout_s=0.001)
        timeout_event = adapter.process_chunk(self.dummy_chunk, event_id=1)

        self.assertEqual(timeout_event.status, "timeout")
        self.assertTrue(timeout_event.session_reset, "session_reset must be True after real runner timeout")
        self.assertIsNone(runner.past_key_values, "Runner cache must be cleared by reset() after timeout")
        self.assertEqual(runner.frames_seen, 0, "Runner frames_seen must be reset to 0")

        # Step 2: Allow standard timeout and verify the runner executes cleanly from fresh state
        adapter.timeout_s = 60.0
        recovered_event = adapter.process_chunk(self.dummy_chunk, event_id=2)

        self.assertEqual(recovered_event.status, "ok", f"Recovered call failed with: {recovered_event.error}")
        self.assertIsNotNone(recovered_event.caption)
        self.assertGreater(len(recovered_event.caption), 0)
        self.assertIsNotNone(runner.past_key_values, "Cache should now contain valid state from recovered chunk")
        adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
