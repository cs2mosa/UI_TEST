from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QCheckBox, QComboBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from app.core.types import VLMEvent
from app.widgets.event_card import EventCard
from app.config import DEFAULT_CONFIG


class EventFeed(QWidget):
    """
    Vertical feed displaying event cards with 'Reveal as played' progressive reveal,
    severity filtering, and auto-scrolling to the active playhead chunk.
    """
    chunk_selected = Signal(int)       # emits start_ms
    verdict_updated = Signal(int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.events: list[VLMEvent] = []
        self.card_widgets: list[EventCard] = []
        self.gt_enabled: bool = False
        
        self.current_ts_ms: int = 0
        self.active_chunk_id: int | None = None
        self.reveal_as_played: bool = True

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)

        # Feed Toolbar: Filter and Reveal Toggle
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(6, 6, 6, 2)

        lbl_title = QLabel("VLM OUTPUT")
        lbl_title.setStyleSheet(f"font-weight: bold; color: {DEFAULT_CONFIG.color_text}; font-size: 11px;")
        top_bar.addWidget(lbl_title)

        top_bar.addStretch()

        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["All", "High Severity (≥0.66)", "Hazards Only", "Errors/Warnings"])
        self.combo_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {DEFAULT_CONFIG.color_surface};
                color: {DEFAULT_CONFIG.color_text};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 10px;
            }}
        """)
        self.combo_filter.currentIndexChanged.connect(self._apply_filter)
        top_bar.addWidget(self.combo_filter)

        self.chk_reveal = QCheckBox("Reveal as played")
        self.chk_reveal.setChecked(True)
        self.chk_reveal.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text_muted}; font-size: 10px;")
        self.chk_reveal.toggled.connect(self._on_reveal_toggled)
        top_bar.addWidget(self.chk_reveal)

        root_layout.addLayout(top_bar)

        # Scroll area for cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {DEFAULT_CONFIG.color_bg};
                border: none;
            }}
        """)

        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(4, 4, 4, 4)
        self.card_layout.setSpacing(6)
        self.card_layout.addStretch()

        self.scroll_area.setWidget(self.card_container)
        root_layout.addWidget(self.scroll_area)

    def set_events(self, events: list[VLMEvent], gt_enabled: bool = False):
        self.events = events
        self.gt_enabled = gt_enabled
        self._rebuild_cards()

    def add_event(self, event: VLMEvent):
        self.events.append(event)
        self._add_single_card(event)

    def clear(self):
        self.events.clear()
        self.active_chunk_id = None
        self._clear_cards()

    def _clear_cards(self):
        for card in self.card_widgets:
            card.setParent(None)
            card.deleteLater()
        self.card_widgets.clear()

    def _add_single_card(self, event: VLMEvent):
        card = EventCard(event, gt_enabled=self.gt_enabled)
        card.clicked.connect(self.chunk_selected.emit)
        card.verdict_changed.connect(self.verdict_updated.emit)
        self.card_widgets.append(card)

        # Insert before stretch
        self.card_layout.insertWidget(self.card_layout.count() - 1, card)
        self._update_card_visibility(card)

    def _rebuild_cards(self):
        self._clear_cards()
        for event in self.events:
            self._add_single_card(event)

    def _on_reveal_toggled(self, checked: bool):
        self.reveal_as_played = checked
        self._apply_filter()

    def set_playhead(self, ts_ms: int, active_chunk_id: int | None):
        self.current_ts_ms = ts_ms
        self.active_chunk_id = active_chunk_id

        # Update active highlight and visibility
        active_card = None
        for card in self.card_widgets:
            is_active = (card.vlm_event.chunk_id == active_chunk_id)
            card.set_active(is_active)
            if is_active:
                active_card = card

            if self.reveal_as_played:
                self._update_card_visibility(card)

        # Auto-scroll active card into view
        if active_card and active_card.isVisible():
            self.scroll_area.ensureWidgetVisible(active_card, 50, 50)

    def _apply_filter(self):
        for card in self.card_widgets:
            self._update_card_visibility(card)

    def _update_card_visibility(self, card: EventCard):
        # 1. Reveal as played rule: only show chunks that ended at or before current playhead
        if self.reveal_as_played:
            if card.vlm_event.end_ms > self.current_ts_ms and self.current_ts_ms > 0:
                card.setVisible(False)
                return

        # 2. Filter rule
        filter_text = self.combo_filter.currentText()
        score = card.vlm_event.score if card.vlm_event.score is not None else 0.0

        if filter_text == "High Severity (≥0.66)":
            card.setVisible(score >= DEFAULT_CONFIG.high_thr)
        elif filter_text == "Hazards Only":
            card.setVisible(score >= 0.3)
        elif filter_text == "Errors/Warnings":
            card.setVisible(card.vlm_event.status in ("error", "timeout", "invalid"))
        else:
            card.setVisible(True)
