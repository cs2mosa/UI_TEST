from __future__ import annotations
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal
from app.config import DEFAULT_CONFIG


class SettingsBar(QFrame):
    """
    Top application toolbar with file loading, chunk settings inputs,
    process/reprocess/cancel controls, and export actions.
    """
    open_clicked = Signal()
    process_clicked = Signal()
    cancel_clicked = Signal()
    export_json_clicked = Signal()
    export_csv_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            SettingsBar {{
                background-color: {DEFAULT_CONFIG.color_surface};
                border-bottom: 1px solid {DEFAULT_CONFIG.color_border};
            }}
            QPushButton {{
                background-color: {DEFAULT_CONFIG.color_surface_alt};
                color: {DEFAULT_CONFIG.color_text};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {DEFAULT_CONFIG.color_primary};
                color: #ffffff;
            }}
            QPushButton:disabled {{
                background-color: {DEFAULT_CONFIG.color_surface};
                color: {DEFAULT_CONFIG.color_text_muted};
                border-color: {DEFAULT_CONFIG.color_surface_alt};
            }}
            QLabel {{
                color: {DEFAULT_CONFIG.color_text};
                font-size: 11px;
            }}
            QDoubleSpinBox, QSpinBox {{
                background-color: {DEFAULT_CONFIG.color_bg};
                color: {DEFAULT_CONFIG.color_text};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
            }}
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # Open Media Button
        self.btn_open = QPushButton("📁 Open Media")
        self.btn_open.clicked.connect(self.open_clicked.emit)
        layout.addWidget(self.btn_open)

        # Process Button
        self.btn_process = QPushButton("⚡ Process")
        self.btn_process.setStyleSheet(f"""
            QPushButton {{
                background-color: {DEFAULT_CONFIG.color_primary};
                color: #ffffff;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DEFAULT_CONFIG.color_primary_hover};
            }}
        """)
        self.btn_process.clicked.connect(self.process_clicked.emit)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)

        # Cancel Button
        self.btn_cancel = QPushButton("✖ Cancel")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.btn_cancel.clicked.connect(self.cancel_clicked.emit)
        self.btn_cancel.setVisible(False)
        layout.addWidget(self.btn_cancel)

        # Separator
        layout.addSpacing(6)

        # Chunk duration
        layout.addWidget(QLabel("Chunk:"))
        self.spin_chunk = QDoubleSpinBox()
        self.spin_chunk.setRange(1.0, 60.0)
        self.spin_chunk.setSingleStep(0.5)
        self.spin_chunk.setValue(DEFAULT_CONFIG.default_chunk_s)
        self.spin_chunk.setSuffix("s")
        layout.addWidget(self.spin_chunk)

        # Overlap duration
        layout.addWidget(QLabel("Overlap:"))
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 30.0)
        self.spin_overlap.setSingleStep(0.5)
        self.spin_overlap.setValue(DEFAULT_CONFIG.default_overlap_s)
        self.spin_overlap.setSuffix("s")
        layout.addWidget(self.spin_overlap)

        # Sampling FPS
        layout.addWidget(QLabel("VLM fps:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 30)
        self.spin_fps.setValue(int(DEFAULT_CONFIG.default_vlm_fps))
        layout.addWidget(self.spin_fps)

        # Ground truth status badge
        self.lbl_gt_badge = QLabel("Ground Truth: ✔ scene.gt.csv")
        self.lbl_gt_badge.setStyleSheet("color: #10b981; font-weight: bold; padding: 2px 6px; background-color: rgba(16, 185, 129, 0.15); border-radius: 3px;")
        self.lbl_gt_badge.setVisible(False)
        layout.addWidget(self.lbl_gt_badge)

        layout.addStretch()

        # In-progress processing indicators
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {DEFAULT_CONFIG.color_bg};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {DEFAULT_CONFIG.color_primary};
                border-radius: 2px;
            }}
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.lbl_progress_eta = QLabel("")
        self.lbl_progress_eta.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text_muted}; font-size: 10px;")
        self.lbl_progress_eta.setVisible(False)
        layout.addWidget(self.lbl_progress_eta)

        # Export buttons
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.clicked.connect(self.export_json_clicked.emit)
        self.btn_export_json.setEnabled(False)
        layout.addWidget(self.btn_export_json)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self.export_csv_clicked.emit)
        self.btn_export_csv.setEnabled(False)
        layout.addWidget(self.btn_export_csv)

    def set_ground_truth(self, filename: str | None):
        if filename:
            self.lbl_gt_badge.setText(f"Ground Truth: ✔ {filename}")
            self.lbl_gt_badge.setVisible(True)
        else:
            self.lbl_gt_badge.setVisible(False)

    def set_state_loaded(self):
        self.btn_process.setText("⚡ Process")
        self.btn_process.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.spin_chunk.setEnabled(True)
        self.spin_overlap.setEnabled(True)
        self.spin_fps.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_eta.setVisible(False)

    def set_state_processing(self):
        self.btn_process.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.spin_chunk.setEnabled(False)
        self.spin_overlap.setEnabled(False)
        self.spin_fps.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_eta.setVisible(True)
        self.lbl_progress_eta.setText("Starting VLM analysis...")

    def update_progress(self, completed: int, total: int, eta_s: float):
        if total > 0:
            pct = int((completed / total) * 100)
            self.progress_bar.setValue(pct)
            self.lbl_progress_eta.setText(f"{completed}/{total} chunks · ETA {eta_s:.1f}s")

    def set_state_ready(self, has_results: bool = True):
        self.btn_process.setText("⚡ Reprocess")
        self.btn_process.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.spin_chunk.setEnabled(True)
        self.spin_overlap.setEnabled(True)
        self.spin_fps.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_eta.setVisible(False)
        self.btn_export_json.setEnabled(has_results)
        self.btn_export_csv.setEnabled(has_results)
