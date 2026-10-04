import os
import unittest
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from app.config import DEFAULT_CONFIG
from examples.mock_vlm import MockVLM

# Ensure a single QApplication instance exists
app = QApplication.instance() or QApplication([])

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_vlm = MockVLM(simulated_delay_s=0.01)
        self.window = MainWindow(vlm=self.mock_vlm, model_label="test-runner", config=DEFAULT_CONFIG)
        self.video_path = r"F:\URCA_PROJECTS\vlm_test_ui\examples\synthetic_test.mp4"

    def tearDown(self):
        self.window.close()

    def test_state_flow(self):
        # 1. EMPTY state
        self.assertEqual(self.window.current_state, "EMPTY")

        # 2. LOADED state
        self.window.load_media(self.video_path)
        self.assertEqual(self.window.current_state, "LOADED")
        self.assertGreater(self.window.session.duration_ms, 0)
        self.assertTrue(self.window.settings_bar.btn_process.isEnabled())

        # 3. Synchronous chunking validation
        self.window._start_processing()
        self.assertEqual(self.window.current_state, "PROCESSING")

        # Wait for worker thread to finish
        self.window.batch_worker.wait(10000)
        # Process pending Qt events
        app.processEvents()

        # 4. READY state
        self.assertEqual(self.window.current_state, "READY")
        self.assertGreater(len(self.window.session.events), 0)

        # 5. Active chunk playhead lookup
        active_ev = self.window.session.get_active_event_at_ts(2000)
        self.assertIsNotNone(active_ev)
        self.assertEqual(active_ev.status, "ok")

        # 6. Summary metrics
        m = self.window.session.get_summary_metrics()
        self.assertEqual(m["total_chunks"], len(self.window.session.events))

if __name__ == "__main__":
    unittest.main()
