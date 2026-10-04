from __future__ import annotations
import sys
import os
import argparse

# Ensure vlm_test_ui directory and root are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from app.config import DEFAULT_CONFIG
from examples.mock_vlm import MockVLM


def main():
    parser = argparse.ArgumentParser(description="VLM Safety Monitor Desktop Test UI")
    parser.add_argument("--qwen", action="store_true", help="Load quantized Qwen-VL model")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG.qwen_model_path, help="HuggingFace model path/identifier")
    parser.add_argument("--delay", type=float, default=0.25, help="Simulated delay per chunk for mock VLM (seconds)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("VLM Safety Monitor")

    if args.qwen:
        try:
            from vlm.qwen3vl_runner import Qwen3VLRunner
            print(f"[*] Initializing Qwen-VL runner from {args.model_path}...")
            runner = Qwen3VLRunner(model_path=args.model_path)
            model_label = os.path.basename(args.model_path)
        except Exception as e:
            print(f"[!] Could not load Qwen runner ({e}). Falling back to Mock VLM.")
            runner = MockVLM(simulated_delay_s=args.delay)
            model_label = "mock-vlm-safety"
    else:
        print("[*] Running with Mock VLM (use --qwen to load quantized Qwen VL)")
        runner = MockVLM(simulated_delay_s=args.delay)
        model_label = "mock-vlm-safety"

    window = MainWindow(vlm=runner, model_label=model_label, config=DEFAULT_CONFIG)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
