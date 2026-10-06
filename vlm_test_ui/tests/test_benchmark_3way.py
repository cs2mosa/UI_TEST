"""
tests/test_benchmark_3way.py  -  3WAY-1/3WAY-2 dry-run tests for Phase 6.

Verifies compare_latency.py three-way additions with injected fake data.
No GPU, no model, no network.
"""
import importlib.util
import io
import os
import sys
import unittest
import contextlib

# compare_latency.py lives in UI_TEST/, two dirs above vlm_test_ui/tests/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CL_PATH = os.path.join(BASE_DIR, 'compare_latency.py')


def _load_cl():
    spec = importlib.util.spec_from_file_location('compare_latency', _CL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake telemetry helpers
# ---------------------------------------------------------------------------
def _chunk(idx, jev=False):
    c = {
        'chunk_id': idx, 'num_frames': 20,
        'total_time_s': 10.0 + idx,
        'ingest_time_s': 2.0, 'compress_time_s': 1.0, 'gen_time_s': 7.0,
        'latency_per_frame_ms': 500.0,
        'vision_tokens_ingested': 100,
        'kv_len_pre_compress': 200, 'kv_len_post_compress': 100,
        'kv_cache_len': 100, 'peak_vram_mb': 3000.0,
        'caption': 'test', 'score': 0.1,
    }
    if jev:
        c['jev_time_s'] = 0.5
        c['jev_tier'] = 'FULL'
        c['jev_reason'] = 'above_watch'
        c['jev_pmax'] = 0.26
        c['frames_sent'] = 18
        c['frames_full_equiv'] = 20
        c['net_total_time_s'] = c['total_time_s'] - c['jev_time_s']
    return c


def _make_data(engine, jev=False, n=3):
    return {'engine': engine, 'chunks': [_chunk(i, jev=jev) for i in range(n)]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class Test3Way1(unittest.TestCase):
    """3WAY-1: JSON plumbing and net = gross - jev."""

    def setUp(self):
        self.cl = _load_cl()
        self.h  = _make_data('HERMES')
        self.i  = _make_data('InfiniPot-V')
        self.jh = _make_data('JEV+HERMES', jev=True)

    def test_net_equals_gross_minus_jev(self):
        for c in self.jh['chunks']:
            expected = c['total_time_s'] - c['jev_time_s']
            self.assertAlmostEqual(c['net_total_time_s'], expected, places=9)

    def test_all_required_keys_present(self):
        required = [
            'total_time_s', 'ingest_time_s', 'compress_time_s', 'gen_time_s',
            'latency_per_frame_ms', 'kv_cache_len', 'peak_vram_mb',
            'caption', 'score',
            'jev_time_s', 'jev_tier', 'jev_reason', 'jev_pmax',
            'frames_sent', 'frames_full_equiv', 'net_total_time_s',
        ]
        for c in self.jh['chunks']:
            for k in required:
                self.assertIn(k, c, msg='Missing key: ' + k)

    def test_three_column_table_renders(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.cl.print_comparison_table(self.h, self.i, self.jh)
        out = buf.getvalue()
        for marker in ['JEV+HERMES', 'JEV s', 'Net HERMES time', 'JEV+HERMES Frame Econ']:
            self.assertIn(marker, out, msg='3-col table missing: ' + marker)

    def test_existing_two_column_mode_unchanged(self):
        """Backward compat: calling without jh_data still works as before."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.cl.print_comparison_table(self.h, self.i)
        out = buf.getvalue()
        self.assertIn('Speedup / Diff', out)
        self.assertNotIn('JEV+HERMES', out)

    def test_existing_output_filenames_unchanged(self):
        """hermes_benchmark_telemetry.json and infinipot_benchmark_telemetry.json
        are referenced in master() unchanged."""
        import pathlib
        src = pathlib.Path(_CL_PATH).read_text(encoding='utf-8')
        self.assertIn('"hermes_benchmark_telemetry.json"', src)
        self.assertIn('"infinipot_benchmark_telemetry.json"', src)
        self.assertIn('"jev_hermes_benchmark_telemetry.json"', src)


class Test3Way2(unittest.TestCase):
    """3WAY-2: JEV s column and frame economy summary line."""

    def setUp(self):
        self.cl = _load_cl()

    def test_jev_s_column_present_in_header(self):
        h  = _make_data('HERMES')
        i  = _make_data('InfiniPot-V')
        jh = _make_data('JEV+HERMES', jev=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.cl.print_comparison_table(h, i, jh)
        out = buf.getvalue()
        self.assertIn('JEV s', out)

    def test_frame_economy_savings_in_summary(self):
        h  = _make_data('HERMES')
        i  = _make_data('InfiniPot-V')
        jh = _make_data('JEV+HERMES', jev=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.cl.print_comparison_table(h, i, jh)
        out = buf.getvalue()
        # 18/20 per chunk * 3 chunks = 54/60 => 10.0% fewer
        self.assertIn('54/60', out)
        self.assertIn('10.0%', out)


if __name__ == '__main__':
    unittest.main()
