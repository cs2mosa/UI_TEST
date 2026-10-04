from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen, QDragEnterEvent, QDropEvent

from app.core.types import VLMEvent
from app.config import DEFAULT_CONFIG


class VideoView(QWidget):
    """
    Video/Image rendering widget with aspect-ratio scaling, drop-zone support,
    and a semi-transparent HUD overlay showing the active chunk's VLM caption and score.
    """
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(480, 320)
        
        self.current_frame: QPixmap | None = None
        self.active_event: VLMEvent | None = None
        self.is_empty_state: bool = True
        self.gt_enabled: bool = False

        self.setStyleSheet(f"background-color: {DEFAULT_CONFIG.color_bg};")

    def set_empty_state(self, empty: bool):
        self.is_empty_state = empty
        if empty:
            self.current_frame = None
            self.active_event = None
        self.update()

    def set_frame(self, qimage: QImage):
        self.is_empty_state = False
        self.current_frame = QPixmap.fromImage(qimage)
        self.update()

    def set_active_event(self, event: VLMEvent | None, gt_enabled: bool = False):
        self.active_event = event
        self.gt_enabled = gt_enabled
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(DEFAULT_CONFIG.color_bg))

        if self.is_empty_state or self.current_frame is None:
            # Draw Drop Zone Hint
            painter.setPen(QPen(QColor(DEFAULT_CONFIG.color_border), 2, Qt.PenStyle.DashLine))
            hint_rect = self.rect().adjusted(24, 24, -24, -24)
            painter.drawRoundedRect(hint_rect, 12, 12)

            painter.setPen(QColor(DEFAULT_CONFIG.color_text_muted))
            font = QFont("Segoe UI", 13)
            painter.setFont(font)
            painter.drawText(
                hint_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Drag & Drop Media File Here\nor click [Open Media] above\n\nSupports: .mp4, .avi, .mov, .mkv, .jpg, .png",
            )
            return

        # Draw Video Frame scaled with Aspect Ratio
        scaled_pixmap = self.current_frame.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

        # Draw Active Chunk HUD Overlay if available
        if self.active_event and (self.active_event.caption or self.active_event.score is not None):
            self._draw_hud_overlay(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())

    def _draw_hud_overlay(self, painter: QPainter, vx: int, vy: int, vw: int, vh: int):
        score = self.active_event.score if self.active_event.score is not None else 0.0
        caption = self.active_event.caption or "(No caption)"
        
        # Color code based on score severity or GT match
        if score >= DEFAULT_CONFIG.high_thr:
            theme_color = QColor(DEFAULT_CONFIG.color_hazard)
            level_str = "CRITICAL / HIGH"
        elif score >= 0.3:
            theme_color = QColor(DEFAULT_CONFIG.color_warning)
            level_str = "MEDIUM"
        else:
            theme_color = QColor(DEFAULT_CONFIG.color_safe)
            level_str = "LOW / SAFE"

        # HUD Box position inside video area
        padding = 16
        box_w = min(vw - 32, 600)
        box_h = 75 if not (self.gt_enabled and self.active_event.expected_score is not None) else 105
        box_x = vx + (vw - box_w) // 2
        box_y = vy + vh - box_h - padding

        # Draw semi-transparent dark HUD card
        painter.setBrush(QColor(15, 23, 42, 220)) # Dark slate transparent
        painter.setPen(QPen(theme_color, 1.5))
        painter.drawRoundedRect(box_x, box_y, box_w, box_h, 8, 8)

        # Score Badge
        badge_rect_w = 120
        badge_rect_h = 24
        painter.setBrush(theme_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(box_x + 12, box_y + 12, badge_rect_w, badge_rect_h, 4, 4)

        # Badge text
        painter.setPen(QColor("#ffffff"))
        badge_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(badge_font)
        painter.drawText(
            box_x + 12, box_y + 12, badge_rect_w, badge_rect_h,
            Qt.AlignmentFlag.AlignCenter,
            f"VLM: {score:.2f} {level_str[:4]}"
        )

        # Caption text
        painter.setPen(QColor(DEFAULT_CONFIG.color_text))
        caption_font = QFont("Segoe UI", 10)
        painter.setFont(caption_font)
        text_rect_x = box_x + badge_rect_w + 24
        text_rect_w = box_w - badge_rect_w - 36
        painter.drawText(
            text_rect_x, box_y + 10, text_rect_w, 40,
            Qt.TextFlag.TextWordWrap,
            caption
        )

        # GT line if active
        if self.gt_enabled and self.active_event.expected_score is not None:
            gt_score = self.active_event.expected_score
            gt_desc = self.active_event.gt_description or "Safe / Normal"
            match_str = self.active_event.match or "MATCH"
            
            gt_y = box_y + 60
            painter.setPen(QColor(DEFAULT_CONFIG.color_text_muted))
            gt_font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
            painter.setFont(gt_font)
            painter.drawText(box_x + 12, gt_y, f"GT: {gt_score:.2f} ({match_str})  \"{gt_desc}\"")
