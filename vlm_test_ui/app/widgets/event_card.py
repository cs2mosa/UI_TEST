from __future__ import annotations
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QInputDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from app.core.types import VLMEvent
from app.widgets.playback_bar import format_ms
from app.config import DEFAULT_CONFIG


class EventCard(QFrame):
    """
    Card representing a single VLM inference event.
    Displays timestamps, scores, captions, status badges, and tester review actions.
    """
    clicked = Signal(int)             # emits chunk start_ms
    verdict_changed = Signal(int, str, str)  # event_id, verdict, note

    def __init__(self, event: VLMEvent, gt_enabled: bool = False, parent=None):
        super().__init__(parent)
        self.vlm_event = event
        self.gt_enabled = gt_enabled
        self.is_active = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._init_ui()
        self.update_style()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Header Row: Timestamp span + Status/Match Badge
        hdr_layout = QHBoxLayout()
        span_str = f"{format_ms(self.vlm_event.start_ms)} → {format_ms(self.vlm_event.end_ms)}"
        self.lbl_span = QLabel(span_str)
        self.lbl_span.setStyleSheet(f"font-weight: bold; color: {DEFAULT_CONFIG.color_text}; font-size: 11px;")
        hdr_layout.addWidget(self.lbl_span)

        hdr_layout.addStretch()

        # Status / Match Badge
        self.lbl_badge = QLabel()
        self.lbl_badge.setStyleSheet("font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 3px;")
        self._update_badge()
        hdr_layout.addWidget(self.lbl_badge)
        layout.addLayout(hdr_layout)

        # VLM Score Bar
        score_layout = QHBoxLayout()
        score_val = self.vlm_event.score if self.vlm_event.score is not None else 0.0
        lbl_vlm = QLabel(f"VLM {score_val:.2f}")
        lbl_vlm.setStyleSheet(f"font-size: 10px; color: {DEFAULT_CONFIG.color_text_muted}; min-width: 55px;")
        score_layout.addWidget(lbl_vlm)

        self.bar_score = QProgressBar()
        self.bar_score.setRange(0, 100)
        self.bar_score.setValue(int(score_val * 100))
        self.bar_score.setTextVisible(False)
        self.bar_score.setFixedHeight(6)
        
        # Color bar according to score
        if score_val >= DEFAULT_CONFIG.high_thr:
            bar_color = DEFAULT_CONFIG.color_hazard
        elif score_val >= 0.3:
            bar_color = DEFAULT_CONFIG.color_warning
        else:
            bar_color = DEFAULT_CONFIG.color_safe

        self.bar_score.setStyleSheet(f"""
            QProgressBar {{
                background-color: {DEFAULT_CONFIG.color_surface_alt};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {bar_color};
                border-radius: 3px;
            }}
        """)
        score_layout.addWidget(self.bar_score)
        layout.addLayout(score_layout)

        # Ground Truth row if active
        if self.gt_enabled and self.vlm_event.expected_score is not None:
            gt_layout = QHBoxLayout()
            lbl_gt = QLabel(f"GT  {self.vlm_event.expected_score:.2f}")
            lbl_gt.setStyleSheet(f"font-size: 10px; color: {DEFAULT_CONFIG.color_text_muted}; min-width: 55px;")
            gt_layout.addWidget(lbl_gt)

            bar_gt = QProgressBar()
            bar_gt.setRange(0, 100)
            bar_gt.setValue(int(self.vlm_event.expected_score * 100))
            bar_gt.setTextVisible(False)
            bar_gt.setFixedHeight(4)
            bar_gt.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {DEFAULT_CONFIG.color_surface_alt};
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background-color: {DEFAULT_CONFIG.color_match};
                    border-radius: 2px;
                }}
            """)
            gt_layout.addWidget(bar_gt)
            layout.addLayout(gt_layout)

        # Caption / Error Text
        self.lbl_caption = QLabel()
        self.lbl_caption.setWordWrap(True)
        if self.vlm_event.status == "ok":
            self.lbl_caption.setText(self.vlm_event.caption or "(No description generated)")
            self.lbl_caption.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text}; font-size: 11px;")
        elif self.vlm_event.status == "invalid":
            self.lbl_caption.setText(f"⚠ Invalid reply: {self.vlm_event.caption}")
            self.lbl_caption.setStyleSheet("color: #f43f5e; font-size: 11px; font-style: italic;")
        else:
            self.lbl_caption.setText(f"✖ Error: {self.vlm_event.error}")
            self.lbl_caption.setStyleSheet("color: #ef4444; font-size: 11px; font-style: italic;")
        layout.addWidget(self.lbl_caption)

        # Review action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_correct = QPushButton("✔")
        self.btn_correct.setToolTip("Mark as Correct detection")
        self.btn_correct.setFixedSize(26, 22)
        self.btn_correct.clicked.connect(self._on_correct_clicked)
        btn_layout.addWidget(self.btn_correct)

        self.btn_wrong = QPushButton("✖")
        self.btn_wrong.setToolTip("Mark as Wrong / False detection")
        self.btn_wrong.setFixedSize(26, 22)
        self.btn_wrong.clicked.connect(self._on_wrong_clicked)
        btn_layout.addWidget(self.btn_wrong)

        self.btn_note = QPushButton("📝")
        self.btn_note.setToolTip("Add / view tester review note")
        self.btn_note.setFixedSize(26, 22)
        self.btn_note.clicked.connect(self._on_note_clicked)
        btn_layout.addWidget(self.btn_note)

        btn_layout.addStretch()

        if self.vlm_event.latency_s is not None:
            if self.vlm_event.latency_per_frame_ms is not None:
                latency_str = f"{self.vlm_event.latency_s:.2f}s ({self.vlm_event.latency_per_frame_ms:.0f} ms/f)"
            else:
                latency_str = f"{self.vlm_event.latency_s:.2f}s"
            lbl_latency = QLabel(latency_str)
            lbl_latency.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text_muted}; font-size: 10px;")
            btn_layout.addWidget(lbl_latency)

        layout.addLayout(btn_layout)

    def _update_badge(self):
        if self.gt_enabled and self.vlm_event.match:
            match = self.vlm_event.match
            if match == "MATCH":
                self.lbl_badge.setText("MATCH ✔")
                self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_match}; color: #ffffff;")
            elif match == "MISS":
                self.lbl_badge.setText("MISS ✖")
                self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_miss}; color: #ffffff;")
            elif match == "FP":
                self.lbl_badge.setText("FALSE POSITIVE")
                self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_fp}; color: #ffffff;")
            else:
                self.lbl_badge.setText("SCORE OFF")
                self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_off}; color: #000000;")
            return

        # Without GT
        score = self.vlm_event.score if self.vlm_event.score is not None else 0.0
        if score >= DEFAULT_CONFIG.high_thr:
            self.lbl_badge.setText("HIGH HAZARD")
            self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_hazard}; color: #ffffff;")
        elif score >= 0.3:
            self.lbl_badge.setText("MEDIUM")
            self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_warning}; color: #000000;")
        else:
            self.lbl_badge.setText("SAFE")
            self.lbl_badge.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_safe}; color: #ffffff;")

    def set_active(self, active: bool):
        if self.is_active != active:
            self.is_active = active
            self.update_style()

    def update_style(self):
        border_color = DEFAULT_CONFIG.color_primary if self.is_active else DEFAULT_CONFIG.color_border
        border_width = "2px" if self.is_active else "1px"
        bg_color = DEFAULT_CONFIG.color_surface_alt if self.is_active else DEFAULT_CONFIG.color_surface

        self.setStyleSheet(f"""
            EventCard {{
                background-color: {bg_color};
                border: {border_width} solid {border_color};
                border-radius: 6px;
            }}
            EventCard:hover {{
                border-color: {DEFAULT_CONFIG.color_primary_hover};
            }}
            QPushButton {{
                background-color: {DEFAULT_CONFIG.color_surface_alt};
                color: {DEFAULT_CONFIG.color_text};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 3px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {DEFAULT_CONFIG.color_primary};
                color: #ffffff;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.vlm_event.start_ms)

    def _on_correct_clicked(self):
        self.vlm_event.verdict = "correct"
        self.verdict_changed.emit(self.vlm_event.id, "correct", self.vlm_event.note)

    def _on_wrong_clicked(self):
        self.vlm_event.verdict = "wrong"
        self.verdict_changed.emit(self.vlm_event.id, "wrong", self.vlm_event.note)

    def _on_note_clicked(self):
        text, ok = QInputDialog.getMultiLineText(
            self, f"Note for Chunk #{self.vlm_event.chunk_id}",
            "Enter review notes / observations:", self.vlm_event.note
        )
        if ok:
            self.vlm_event.note = text
            self.verdict_changed.emit(self.vlm_event.id, self.vlm_event.verdict or "", text)
