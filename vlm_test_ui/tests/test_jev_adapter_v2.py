from __future__ import annotations
"""
test_jev_adapter_v2.py - Phase B unit tests (design_v3 §10.1).

PREFIX-unit 1..4: split_prompt_suffixes / suffix_position_ids (pure helpers).
ORDER-1..2: the `order` kwarg with a stubbed model/processor — no GPU, no model,
no network. The stub model fires the decoder-layer hooks so the position-capture
mechanism (§5.2 step 3) is exercised end to end.
"""
import unittest
import types

import torch

from app.core.jev_adapter import (
    MARKER,
    OneJevDecider,
    split_prompt_suffixes,
    suffix_position_ids,
)


# ---------------------------------------------------------------------------
# PREFIX-unit 1..4: split_prompt_suffixes
# ---------------------------------------------------------------------------

class TestSplitPromptSuffixes(unittest.TestCase):

    # PREFIX-unit 1: exact common-prefix cut
    def test_prefix_unit_1_exact_cut(self):
        prefix_cut = "<|system|><|user|><video><state>{...}</state>"
        prefix_render = prefix_cut + MARKER + "<|end|>"
        fulls = [
            prefix_cut + "\n\nQuestion: q0\n\nOptions:\nA. yes\nB. no",
            prefix_cut + "\n\nQuestion: q2 variant",
        ]
        suffixes = split_prompt_suffixes(prefix_render, fulls)
        self.assertEqual(len(suffixes), 2)
        self.assertEqual(suffixes[0], "\n\nQuestion: q0\n\nOptions:\nA. yes\nB. no")
        self.assertEqual(suffixes[1], "\n\nQuestion: q2 variant")

    # PREFIX-unit 2: mismatch -> ValueError('prompt prefix mismatch')
    def test_prefix_unit_2_mismatch_raises(self):
        prefix_render = "SHARED-PREFIX" + MARKER
        fulls = ["SHARED-PREFIX suffix ok", "DIFFERENT-PREFIX suffix bad"]
        with self.assertRaises(ValueError) as ctx:
            split_prompt_suffixes(prefix_render, fulls)
        self.assertIn("prompt prefix mismatch", str(ctx.exception))

    def test_prefix_unit_2_missing_marker_raises(self):
        with self.assertRaises(ValueError) as ctx:
            split_prompt_suffixes("no marker here", ["whatever"])
        self.assertIn("prompt prefix mismatch", str(ctx.exception))

    # PREFIX-unit 3: single-render passthrough
    def test_prefix_unit_3_single_render(self):
        prefix_cut = "PREFIX|"
        suffixes = split_prompt_suffixes(
            prefix_cut + MARKER, [prefix_cut + "SUFFIX-TEXT"]
        )
        self.assertEqual(suffixes, ["SUFFIX-TEXT"])

    # PREFIX-unit 4: suffix text equals full[len(prefix_cut):] for every render
    def test_prefix_unit_4_suffix_equality(self):
        prefix_cut = "A" * 37 + "|"
        fulls = [prefix_cut + "x" * k for k in (1, 5, 23)]
        suffixes = split_prompt_suffixes(prefix_cut + MARKER, fulls)
        for full, suffix in zip(fulls, suffixes):
            self.assertEqual(suffix, full[len(prefix_cut):])


# ---------------------------------------------------------------------------
# PREFIX-unit (suffix_position_ids): shape [3, 1, L]; values last + 1 + j
# ---------------------------------------------------------------------------

class TestSuffixPositionIds(unittest.TestCase):

    def test_shape_and_values_flat_last(self):
        last = torch.tensor([7.0, 7.0, 7.0])
        out = suffix_position_ids(last, 4)
        self.assertEqual(tuple(out.shape), (3, 1, 4))
        for a in range(3):
            for j in range(4):
                self.assertEqual(float(out[a, 0, j]), 7.0 + 1.0 + j)

    def test_values_per_axis(self):
        last = torch.tensor([1.0, 2.0, 3.0])
        out = suffix_position_ids(last, 3)
        self.assertEqual(float(out[0, 0, 0]), 2.0)
        self.assertEqual(float(out[1, 0, 2]), 2.0 + 1.0 + 2.0)
        self.assertEqual(float(out[2, 0, 1]), 3.0 + 1.0 + 1.0)

    def test_dtype_and_device_preserved(self):
        last = torch.zeros(3, dtype=torch.float32)
        out = suffix_position_ids(last, 2)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, last.device)


# ---------------------------------------------------------------------------
# ORDER-1..2: stubbed forward (no GPU, no model)
# ---------------------------------------------------------------------------

_ID_A = 10
_ID_B = 20


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.device = "cpu"

    def to(self, device):
        return self


class _FakeCache:
    pass


class _FakeOut:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeLayer:
    def __init__(self):
        self._hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self._hooks.append(hook)

        class _Handle:
            def remove(self):
                if hook in self._outer._hooks:
                    self._outer._hooks.remove(hook)
        handle = _Handle()
        handle._outer = self
        return handle

    def fire(self, position_ids):
        for hook in list(self._hooks):
            hook(self, (), {"position_ids": position_ids})


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = self
        self.full_renders: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        user = messages[1]["content"]
        parts = []
        for part in user:
            parts.append("<video>" if part["type"] == "video" else part["text"])
        body = "[SYS][USER]" + "".join(parts)
        if add_generation_prompt:
            self.full_renders.append(body + "[GEN]")
            return body + "[GEN]"
        return body

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 29) for c in text[:16]]

    def __call__(self, **kwargs):
        return {"input_ids": _FakeTensor((1, 11)), "attention_mask": _FakeTensor((1, 11))}


class _FakeModel:
    def __init__(self, value_a: float, value_b: float):
        self.value_a = value_a
        self.value_b = value_b
        self.layers = [_FakeLayer() for _ in range(4)]
        self.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=self.layers)
        )
        self.prefix_calls = 0
        self.suffix_calls = 0

    def __call__(self, **kwargs):
        # simulate the decoder stack passing mrope position_ids down its layers
        position_ids = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        )
        for layer in self.layers:
            layer.fire(position_ids)
        if kwargs.get("past_key_values") is None:
            self.prefix_calls += 1
            return _FakeOut(logits=torch.zeros((1, 1, 32)), past_key_values=_FakeCache())
        self.suffix_calls += 1
        logits = torch.zeros((1, 1, 32))
        logits[0, -1, _ID_A] = self.value_a
        logits[0, -1, _ID_B] = self.value_b
        return _FakeOut(logits=logits)


def _make_stub_decider(value_a: float, value_b: float) -> OneJevDecider:
    """OneJevDecider without __init__ (no model load); shared mode."""
    dec = OneJevDecider.__new__(OneJevDecider)
    dec._processor = _FakeProcessor()
    dec._model = _FakeModel(value_a, value_b)
    dec._id_a = _ID_A
    dec._id_b = _ID_B
    dec.mode = "shared"
    return dec


class TestOrderKwarg(unittest.TestCase):

    # ORDER-1: symmetric stub -> p_yes invariant under the swap; rendering swapped
    def test_order_1_symmetric_invariant(self):
        dec = _make_stub_decider(value_a=1.0, value_b=1.0)
        frames = [torch.randint(0, 255, (8, 8, 3), dtype=torch.uint8).numpy()]
        p_ab = dec(frames, 4.0, 0, order="ab")
        p_ba = dec(frames, 4.0, 0, order="ba")
        self.assertEqual(dec._model.prefix_calls, 2)      # one prefill per call
        self.assertEqual(dec._model.suffix_calls, 10)     # five branches per call
        for hazard in p_ab:
            self.assertAlmostEqual(p_ab[hazard], 0.5, places=6)
            self.assertAlmostEqual(p_ab[hazard], p_ba[hazard], places=6)

    def test_order_1_ba_rendering_swapped(self):
        dec = _make_stub_decider(value_a=1.0, value_b=1.0)
        dec([], 4.0, 0, order="ba")
        render = dec._processor.full_renders[-1]
        self.assertIn("\nA. no: the statement is false / the answer is no\nB. yes: ", render)
        ab_render = _make_stub_decider(1.0, 1.0)
        ab_render([], 4.0, 0, order="ab")
        ab_text = ab_render._processor.full_renders[-1]
        self.assertIn("\nA. yes: ", ab_text)
        self.assertLess(ab_text.index("\nA. yes: "), ab_text.index("\nB. no: "))

    # ORDER-2: asymmetric stub -> p_yes differs and follows the slot mapping
    def test_order_2_asymmetric_differs(self):
        dec = _make_stub_decider(value_a=2.0, value_b=0.0)
        p_ab = dec([], 4.0, 0, order="ab")
        p_ba = dec([], 4.0, 0, order="ba")
        # p_yes(ab) = softmax([2, 0])[0]; p_yes(ba) reads slots as [B, A] -> softmax([0, 2])[0]
        expected_ab = float(torch.exp(torch.tensor(2.0)) / (torch.exp(torch.tensor(2.0)) + 1.0))
        for hazard in p_ab:
            self.assertAlmostEqual(p_ab[hazard], expected_ab, places=5)
            self.assertAlmostEqual(p_ba[hazard], 1.0 - expected_ab, places=5)
            self.assertNotAlmostEqual(p_ab[hazard], p_ba[hazard], places=3)
            self.assertAlmostEqual(p_ab[hazard] + p_ba[hazard], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
