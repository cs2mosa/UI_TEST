"""
GPU integration tests for Qwen3VLHermes engine.

Requires:
  - CUDA GPU with ≥6 GB VRAM
  - Qwen3-VL-4B-Instruct downloaded at DEFAULT_CONFIG.hermes_model_path
  - transformers ≥ 4.57, bitsandbytes, torch ≥ 2.9

Run with:
  python -m pytest tests/test_hermes_engine.py -v --tb=short -x

All tests are marked @pytest.mark.gpu and are skipped if CUDA is unavailable.
"""
import os
import unittest
import pytest
import torch
import numpy as np

# Point HF to the local cache so no network calls are made
os.environ.setdefault("HF_HOME", r"F:\URCA_PROJECTS\.hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

gpu_available = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for HERMES engine tests"
)

# Local model path — already downloaded to the project HF cache
MODEL_PATH = (
    r"F:\URCA_PROJECTS\.hf_cache\hub"
    r"\models--Qwen--Qwen3-VL-4B-Instruct"
    r"\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
KV_SIZE = 2000


class TestHermesEngineLoading(unittest.TestCase):
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA GPU required - skipping HermesEngine tests")
        from vlm.hermes_engine import load_hermes_model
        cls.model, cls.processor = load_hermes_model(MODEL_PATH, kv_size=KV_SIZE)

    @classmethod
    def tearDownClass(cls):
        if cls.model is not None:
            del cls.model
            cls.model = None
        if cls.processor is not None:
            del cls.processor
            cls.processor = None
        torch.cuda.empty_cache()

    def setUp(self):
        if self.model is not None:
            self.model.clear_cache()

    @gpu_available
    def test_load_hermes_model(self):
        """Model loads, returns correct types, and has expected config."""
        from vlm.hermes_engine import Qwen3VLHermes
        model, processor = self.model, self.processor
        self.assertIsInstance(model, Qwen3VLHermes)
        self.assertIsNotNone(processor)
        self.assertEqual(model.num_layers, 36)
        self.assertEqual(model._mrope_section, (24, 20, 20))
        self.assertEqual(model.kv_size, KV_SIZE)
        self.assertFalse(model.training)  # model.eval() was called

    @gpu_available
    def test_encode_init_prompt(self):
        """KV cache is populated after encode_init_prompt."""
        model = self.model
        self.assertIsNone(model.kv_cache)
        model.encode_init_prompt()
        self.assertIsNotNone(model.kv_cache)
        k_layer0 = model.kv_cache[0][0]
        # Shape: [batch=1, num_kv_heads, seq_len, head_dim]
        self.assertEqual(k_layer0.dim(), 4)
        self.assertEqual(k_layer0.shape[0], 1)
        self.assertGreater(k_layer0.shape[2], 0)  # seq_len > 0

    @gpu_available
    def test_encode_single_frame_batch(self):
        """Encoding 2 frames grows the KV cache."""
        model = self.model
        model.encode_init_prompt()
        seq_len_before = model.kv_cache[0][0].shape[2]

        frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        video_tensor = torch.from_numpy(frames)
        model.encode_video_chunk(video_tensor)

        seq_len_after = model.kv_cache[0][0].shape[2]
        self.assertGreater(seq_len_after, seq_len_before)
        self.assertEqual(model.total_processed_frames, 2)

    @gpu_available
    def test_predict_and_compress(self):
        """predict_and_compress does not crash and KV cache stays <= kv_size + visual_start_idx."""
        model = self.model
        model.encode_init_prompt()

        frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        video_tensor = torch.from_numpy(frames)
        model.encode_video_chunk(video_tensor)

        model.predict_and_compress()

        seq_len = model.kv_cache[0][0].shape[2]
        max_allowed = KV_SIZE + model.visual_start_idx + 10  # small slack
        self.assertLessEqual(seq_len, max_allowed)

    @gpu_available
    def test_question_answering_returns_string(self):
        """question_answering generates a non-empty string."""
        model = self.model
        model.encode_init_prompt()

        frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        model.encode_video_chunk(torch.from_numpy(frames))

        prompt = 'Describe any hazards. Reply with JSON only: {"caption": "...", "score": 0.0}'
        input_text = {"question": prompt, "prompt": model.get_prompt(prompt)}
        answer = model.question_answering(input_text, max_new_tokens=60)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 0)

    @gpu_available
    def test_gpu_memory_under_6gb(self):
        """Full cycle: load -> encode 8 frames -> compress -> answer stays < 6 GB."""
        model = self.model
        model.encode_init_prompt()

        frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
        model.encode_video_chunk(torch.from_numpy(frames))
        model.predict_and_compress()

        prompt = '{"caption": "test", "score": 0.0}'
        input_text = {"question": prompt, "prompt": model.get_prompt(prompt)}
        model.question_answering(input_text, max_new_tokens=20)

        used_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.assertLess(used_gb, 6.0, f"GPU memory {used_gb:.2f} GB exceeds 6 GB budget")


if __name__ == "__main__":
    unittest.main()
