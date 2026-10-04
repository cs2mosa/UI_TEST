from __future__ import annotations
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from app.core.types import VLMEvent
from app.config import DEFAULT_CONFIG


class ChunkTimeline(QWidget):
    """
    Two-lane timeline displaying chunk spans, playhead tracking,
    color-coded match or severity levels, and interactive seeking.
    """
    chunk_clicked = Signal(int)    # chunk start_ms
    seek_requested = Signal(int)   # timestamp ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMaximumHeight(80)

        self.duration_ms: int = 0
        self.current_ts_ms: int = 0
        self.events: list[VLMEvent] = []
        self.gt_enabled: bool = False
        self.active_chunk_id: int | None = None
        self.total_expected_chunks: int = 0

    def set_duration(self, duration_ms: int):
        self.duration_ms = max(1, duration_ms)
        self.update()

    def set_playhead(self, ts_ms: int):
        self.current_ts_ms = max(0, min(self.duration_ms, ts_ms))
        self.update()

    def set_events(self, events: list[VLMEvent], gt_enabled: bool = False, total_chunks: int = 0):
        self.events = events
        self.gt_enabled = gt_enabled
        self.total_expected_chunks = total_chunks
        self.update()

    def set_active_chunk(self, chunk_id: int | None):
        self.active_chunk_id = chunk_id
        self.update()

    def _ms_to_x(self, ms: int) -> float:
        if self.duration_ms <= 0:
            return 0.0
        padding = 10.0
        usable_w = float(self.width()) - (padding * 2.0)
        return padding + (float(ms) / float(self.duration_ms)) * usable_w

    def _x_to_ms(self, x: float) -> int:
        if self.duration_ms <= 0:
            return 0
        padding = 10.0
        usable_w = float(self.width()) - (padding * 2.0)
        ratio = max(0.0, min(1.0, (x - padding) / usable_w))
        return int(round(ratio * self.duration_ms))

    def _get_chunk_color(self, event: VLMEvent) -> QColor:
        if event.status in ("error", "timeout"):
            return QColor("#64748b")  # slate-500
        if event.status == "invalid":
            return QColor("#f43f5e")  # rose-500

        if self.gt_enabled and event.match:
            match_map = {
                "MATCH": DEFAULT_CONFIG.color_match,
                "MISS": DEFAULT_CONFIG.color_miss,
                "FP": DEFAULT_CONFIG.color_fp,
                "OFF": DEFAULT_CONFIG.color_off,
            }
            return QColor(match_map.get(event.match, DEFAULT_CONFIG.color_surface_alt))

        # Default: color by VLM score severity
        score = event.score if event.score is not None else 0.0
        if score >= DEFAULT_CONFIG.high_thr:
            return QColor(DEFAULT_CONFIG.color_hazard)
        elif score >= 0.3:
            return QColor(DEFAULT_CONFIG.color_warning)
        else:
            return QColor(DEFAULT_CONFIG.color_safe)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.duration_ms > 0:
            click_x = event.position().x()
            click_ms = self._x_to_ms(click_x)
            
            # Check if clicked on a specific chunk
            for ev in self.events:
                if ev.start_ms <= click_ms <= ev.end_ms:
                    self.chunk_clicked.emit(ev.start_ms)
                    self.seek_requested.emit(ev.start_ms)
                    return
            
            # Otherwise seek to clicked timestamp
            self.seek_requested.emit(click_ms)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw timeline background track
        painter.fillRect(self.rect(), QColor(DEFAULT_CONFIG.color_surface))
        painter.setPen(QColor(DEFAULT_CONFIG.color_border))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if self.duration_ms <= 0:
            return

        lane_h = 16.0
        y_lane0 = 8.0
        y_lane1 = 28.0

        # Draw unprocessed regions if cancelled or partial
        if self.events and self.total_expected_chunks > len(self.events):
            last_end = max(e.end_ms for e in self.events)
            if last_end < self.duration_ms:
                hx = self._ms_to_x(last_end)
                hw = self._ms_to_x(self.duration_ms) - hx
                hatch_rect = QRectF(hx, y_lane0, hw, y_lane1 + lane_h - y_lane0)
                hatch_brush = QBrush(QColor(100, 116, 139, 120), Qt.BrushStyle.DiagCrossPattern)
                painter.fillRect(hatch_rect, hatch_brush)

        # Draw chunk spans
        for ev in self.events:
            lane_y = y_lane0 if (ev.chunk_id % 2 == 0) else y_lane1
            x_start = self._ms_to_x(ev.start_ms)
            x_end = self._ms_to_x(ev.end_ms)
            span_w = max(3.0, x_end - x_start)

            chunk_rect = QRectF(x_start, lane_y, span_w, lane_h)
            color = self._get_chunk_color(ev)
            
            # Active chunk highlight
            is_active = (self.active_chunk_id == ev.chunk_id)
            if is_active:
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
            else:
                painter.setPen(QPen(color.darker(130), 1.0))

            painter.setBrush(color)
            painter.drawRoundedRect(chunk_rect, 3, 3)

        # Draw Playhead cursor
        playhead_x = self._ms_to_x(self.current_ts_ms)
        painter.setPen(QPen(QColor("#ffffff"), 2.0))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), self.height())
        
        # Playhead handle marker
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(int(playhead_x) - 4, int(y_lane0) - 4, 8, 8)
