from __future__ import annotations
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt
from app.models.session_model import SessionModel
from app.config import DEFAULT_CONFIG


class SummaryPanel(QFrame):
    """
    Top-right panel providing real-time aggregated metrics for the current test session:
    chunk count, max severity, latency averages, and GT precision/recall when active.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            SummaryPanel {{
                background-color: {DEFAULT_CONFIG.color_surface};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 6px;
            }}
            QLabel {{
                color: {DEFAULT_CONFIG.color_text};
                font-size: 11px;
            }}
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Title
        lbl_title = QLabel("SESSION SUMMARY")
        lbl_title.setStyleSheet(f"font-weight: bold; color: {DEFAULT_CONFIG.color_text_muted}; font-size: 11px; letter-spacing: 0.5px;")
        layout.addWidget(lbl_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)

        # Chunks count
        self.lbl_chunks = QLabel("Chunks: 0")
        grid.addWidget(self.lbl_chunks, 0, 0)

        # Max score
        self.lbl_max_score = QLabel("Max Score: 0.00")
        grid.addWidget(self.lbl_max_score, 0, 1)

        # Avg score
        self.lbl_avg_score = QLabel("Avg Score: 0.00")
        grid.addWidget(self.lbl_avg_score, 1, 0)

        # High hazards
        self.lbl_high = QLabel("High (≥0.66): 0")
        self.lbl_high.setStyleSheet(f"color: {DEFAULT_CONFIG.color_hazard}; font-weight: bold;")
        grid.addWidget(self.lbl_high, 1, 1)

        # Avg Latency
        self.lbl_latency = QLabel("Avg Latency: 0.00s")
        grid.addWidget(self.lbl_latency, 2, 0)

        # Errors
        self.lbl_errors = QLabel("Errors: 0")
        grid.addWidget(self.lbl_errors, 2, 1)

        # Per-frame Latency & Speed
        self.lbl_frame_latency = QLabel("Lat/Frame: 0.0ms")
        grid.addWidget(self.lbl_frame_latency, 3, 0)

        self.lbl_fps = QLabel("Speed: 0.0 fps")
        self.lbl_fps.setStyleSheet(f"color: {DEFAULT_CONFIG.color_primary}; font-weight: bold;")
        grid.addWidget(self.lbl_fps, 3, 1)

        layout.addLayout(grid)

        # Ground truth section (hidden unless GT is active)
        self.gt_container = QFrame()
        gt_layout = QVBoxLayout(self.gt_container)
        gt_layout.setContentsMargins(0, 4, 0, 0)
        gt_layout.setSpacing(4)

        self.lbl_gt_status = QLabel("GT: Matches 0 · Miss 0 · FP 0")
        self.lbl_gt_status.setStyleSheet("font-weight: bold; color: #10b981;")
        gt_layout.addWidget(self.lbl_gt_status)

        self.lbl_gt_metrics = QLabel("Precision: 0.00 · Recall: 0.00 · MAE: 0.00")
        self.lbl_gt_metrics.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text_muted};")
        gt_layout.addWidget(self.lbl_gt_metrics)

        layout.addWidget(self.gt_container)
        self.gt_container.setVisible(False)

    def update_metrics(self, session: SessionModel):
        m = session.get_summary_metrics()

        self.lbl_chunks.setText(f"Chunks: {m['total_chunks']}")
        self.lbl_max_score.setText(f"Max Score: {m['max_score']:.2f}")
        self.lbl_avg_score.setText(f"Avg Score: {m['avg_score']:.2f}")
        self.lbl_high.setText(f"High (≥0.66): {m['high_severity_count']}")
        self.lbl_latency.setText(f"Avg Latency: {m['avg_latency_s']:.2f}s")
        self.lbl_errors.setText(f"Errors: {m['error_count']}")
        self.lbl_frame_latency.setText(f"Lat/Frame: {m.get('avg_latency_per_frame_ms', 0.0):.1f}ms")
        self.lbl_fps.setText(f"Speed: {m.get('fps', 0.0):.1f} fps")

        if session.gt_enabled:
            self.gt_container.setVisible(True)
            # Calculate match metrics if GT is active
            matches = sum(1 for e in session.events if e.match == "MATCH")
            misses = sum(1 for e in session.events if e.match == "MISS")
            fps = sum(1 for e in session.events if e.match == "FP")
            self.lbl_gt_status.setText(f"GT: Matches {matches} · Misses {misses} · FP {fps}")
        else:
            self.gt_container.setVisible(False)
