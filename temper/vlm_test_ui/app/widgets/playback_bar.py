from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox
)
from PySide6.QtCore import Qt, Signal
from app.config import DEFAULT_CONFIG


def format_ms(ms: int) -> str:
    """Formats milliseconds into mm:ss.d"""
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    tenths = int((total_seconds * 10) % 10)
    return f"{minutes:02d}:{seconds:02d}.{tenths}"


class PlaybackBar(QWidget):
    """
    Bottom control bar for media playback: play/pause, time progress slider,
    timestamp readout, and playback speed dropdown.
    """
    play_toggled = Signal(bool)          # True = play, False = pause
    seek_requested = Signal(int)         # timestamp in ms
    speed_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration_ms = 0
        self.is_playing = False
        self._user_is_scrubbing = False

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # Play/Pause Button
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background-color: {DEFAULT_CONFIG.color_primary};
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {DEFAULT_CONFIG.color_primary_hover};
            }}
            QPushButton:disabled {{
                background-color: {DEFAULT_CONFIG.color_surface_alt};
                color: {DEFAULT_CONFIG.color_text_muted};
            }}
        """)
        self.btn_play.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.btn_play)

        # Time slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {DEFAULT_CONFIG.color_surface_alt};
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {DEFAULT_CONFIG.color_primary};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #ffffff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {DEFAULT_CONFIG.color_primary};
            }}
        """)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.slider)

        # Time readout label
        self.lbl_time = QLabel("00:00.0 / 00:00.0")
        self.lbl_time.setStyleSheet(f"color: {DEFAULT_CONFIG.color_text_muted}; font-size: 12px; font-family: monospace;")
        layout.addWidget(self.lbl_time)

        # Speed Dropdown
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.setStyleSheet(f"""
            QComboBox {{
                background-color: {DEFAULT_CONFIG.color_surface};
                color: {DEFAULT_CONFIG.color_text};
                border: 1px solid {DEFAULT_CONFIG.color_border};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 12px;
            }}
        """)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self.combo_speed)

    def set_duration(self, duration_ms: int):
        self.duration_ms = max(0, duration_ms)
        self.slider.setRange(0, self.duration_ms)
        self.update_timestamp(0)

    def update_timestamp(self, ts_ms: int):
        if not self._user_is_scrubbing:
            self.slider.blockSignals(True)
            self.slider.setValue(ts_ms)
            self.slider.blockSignals(False)

        current_str = format_ms(ts_ms)
        total_str = format_ms(self.duration_ms)
        self.lbl_time.setText(f"{current_str} / {total_str}")

    def set_playing_state(self, playing: bool):
        self.is_playing = playing
        self.btn_play.setText("⏸" if playing else "▶")

    def _on_play_clicked(self):
        new_state = not self.is_playing
        self.set_playing_state(new_state)
        self.play_toggled.emit(new_state)

    def _on_slider_pressed(self):
        self._user_is_scrubbing = True

    def _on_slider_released(self):
        self._user_is_scrubbing = False
        self.seek_requested.emit(self.slider.value())

    def _on_slider_moved(self, value: int):
        if self._user_is_scrubbing:
            self.lbl_time.setText(f"{format_ms(value)} / {format_ms(self.duration_ms)}")

    def _on_speed_changed(self):
        text = self.combo_speed.currentText().replace("x", "")
        try:
            speed = float(text)
            self.speed_changed.emit(speed)
        except ValueError:
            pass
