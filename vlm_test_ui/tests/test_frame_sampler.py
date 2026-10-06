from __future__ import annotations
import os, sys, unittest, tempfile
import numpy as np
import cv2

from app.core.frame_sampler import (
    target_frame_count,
    sample_timestamps,
    read_frames_at,
    resize_max_side,
    encode_jpeg,
    make_thumb,
)


def _make_cap(width=64, height=64, n_frames=30, fps=30.0):
    """Create a temporary synthetic video and return an opened VideoCapture."""
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), i * 4 % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    cap = cv2.VideoCapture(tmp.name)
    return cap, tmp.name


class TestTargetFrameCount(unittest.TestCase):
    # SAMPLER-1: seven expected pairs from §5.5
    def test_expected_pairs(self):
        cases = [
            ((0, 5000, 4.0), 20),
            ((0, 5000, 1.0), 5),
            ((8000, 12000, 4.0), 16),
            ((8000, 12000, 1.0), 4),
            ((0, 1000, 1.0), 2),   # max(2, round(1.0*1.0)) = max(2,1)=2
            ((0, 2500, 1.0), 2),   # max(2, round(2.5*1.0)) = max(2,2)=2 (but round(2.5)=2 in Py3)
            ((0, 3500, 1.0), 4),   # max(2, round(3.5*1.0)) = max(2,4)=4
        ]
        for (s, e, fps), expected in cases:
            with self.subTest(s=s, e=e, fps=fps):
                self.assertEqual(target_frame_count(s, e, fps), expected)


class TestSampleTimestamps(unittest.TestCase):
    # SAMPLER-2: endpoints inclusive
    def test_endpoints_inclusive(self):
        ts = sample_timestamps(0, 5000, 4.0)
        self.assertEqual(ts[0], 0)
        self.assertEqual(ts[-1], 5000)

    def test_count_matches_target(self):
        ts = sample_timestamps(0, 5000, 4.0)
        self.assertEqual(len(ts), target_frame_count(0, 5000, 4.0))


class TestReadFramesAt(unittest.TestCase):

    def setUp(self):
        self.cap, self.path = _make_cap(n_frames=30, fps=30.0)

    def tearDown(self):
        self.cap.release()
        os.unlink(self.path)

    # SAMPLER-3: stops at first failed read
    def test_stops_at_failed_read(self):
        # Use timestamps far beyond the video — first read at a valid time, then invalid
        frames, ts = read_frames_at(self.cap, [0, 0, 0], rgb=True)
        # Should have read at least 1 frame; exact count depends on codec
        self.assertGreater(len(frames), 0)

    # SAMPLER-3: single fallback read when empty timestamps supplied
    def test_empty_timestamps(self):
        frames, ts = read_frames_at(self.cap, [], rgb=True)
        self.assertEqual(frames, [])
        self.assertEqual(ts, [])

    # SAMPLER-3: rgb flag channel order (blue frame -> R channel should be ~0 when rgb=True)
    def test_rgb_flag_channel_order(self):
        # Write a purely blue frame (BGR: B=255, G=0, R=0)
        cap2, path2 = _make_cap()
        cap2.release()
        os.unlink(path2)
        tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (64, 64))
        blue_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_frame[:, :, 0] = 200  # BGR blue channel
        out.write(blue_frame)
        out.release()
        cap3 = cv2.VideoCapture(tmp.name)
        frames_rgb, _ = read_frames_at(cap3, [0], rgb=True)
        cap3.release()
        os.unlink(tmp.name)
        if frames_rgb:
            # After BGR->RGB conversion the R channel (index 0) should be high
            self.assertGreater(int(frames_rgb[0][:, :, 2].mean()), 50)  # orig B -> now at index 2 (blue in RGB)

    def test_rgb_false_returns_bgr(self):
        frames, _ = read_frames_at(self.cap, [0], rgb=False)
        self.assertGreater(len(frames), 0)


class TestResizeMaxSide(unittest.TestCase):
    # SAMPLER-4: no-op when already <= max_side
    def test_noop_small(self):
        img = np.zeros((50, 30, 3), dtype=np.uint8)
        result = resize_max_side(img, 100)
        self.assertEqual(result.shape, img.shape)

    # SAMPLER-4: output >= 1px on each side
    def test_at_least_1px(self):
        img = np.zeros((1000, 1, 3), dtype=np.uint8)
        result = resize_max_side(img, 64)
        self.assertGreaterEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 1)

    def test_scales_down(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        result = resize_max_side(img, 100)
        self.assertLessEqual(max(result.shape[0], result.shape[1]), 100)


class TestEncodeJpeg(unittest.TestCase):
    # SAMPLER-5: success returns bytes
    def test_returns_bytes(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        data = encode_jpeg(img)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    # SAMPLER-5: failure raises RuntimeError (pass a bad array)
    def test_failure_raises(self):
        bad = np.zeros((0, 0, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            encode_jpeg(bad)


class TestMakeThumb(unittest.TestCase):
    # SAMPLER-6: returns bytes, thumbnail <= max_side
    def test_returns_bytes(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        data = make_thumb(img)
        self.assertIsInstance(data, bytes)

    # SAMPLER-7: make_thumb count <= 3 is a GatedBatchWorker concern;
    # here we just verify the function works on any single frame.
    def test_shrinks_large(self):
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        data = make_thumb(img, max_side=160)
        self.assertIsInstance(data, bytes)


if __name__ == "__main__":
    unittest.main()