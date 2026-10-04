from __future__ import annotations
import os
import time
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QStatusBar
)
from PySide6.QtCore import Qt, QKeyCombination
from PySide6.QtGui import QKeySequence, QShortcut, QIcon

from app.core.types import VLMCallable, VLMEvent
from app.core.vlm_adapter import VLMAdapter
from app.models.session_model import SessionModel
from app.workers.player_worker import PlayerWorker
from app.workers.batch_worker import BatchWorker
from app.widgets.video_view import VideoView
from app.widgets.playback_bar import PlaybackBar
from app.widgets.chunk_timeline import ChunkTimeline
from app.widgets.event_feed import EventFeed
from app.widgets.summary_panel import SummaryPanel
from app.widgets.settings_bar import SettingsBar
from app.io.exporter import SessionExporter
from app.config import DEFAULT_CONFIG, AppConfig


class MainWindow(QMainWindow):
    """
    Main Desktop Window for the VLM Safety Monitor Test UI.
    Orchestrates the two-phase workflow (Process -> Inspect), synchronized playback,
    timeline interaction, and export facilities.
    """
    def __init__(
        self,
        vlm: VLMCallable,
        model_label: str = "quantized-qwen-vl",
        config: AppConfig = DEFAULT_CONFIG,
        parent=None,
    ):
        super().__init__(parent)
        self.vlm = vlm
        self.model_label = model_label
        self.config = config
        
        self.session = SessionModel()
        self.adapter = VLMAdapter(self.vlm)
        self.player_worker = PlayerWorker(self)
        self.batch_worker: Optional[BatchWorker] = None

        self.current_state = "EMPTY"  # "EMPTY" | "LOADED" | "PROCESSING" | "READY"
        self.total_planned_chunks = 0

        self.setWindowTitle(f"VLM Safety Monitor — [{self.model_label}]")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        self._init_ui()
        self._init_shortcuts()
        self._connect_signals()

        # Start background player thread
        self.player_worker.start()

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setStyleSheet(f"background-color: {self.config.color_bg};")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Settings & Control Bar
        self.settings_bar = SettingsBar(self)
        root_layout.addWidget(self.settings_bar)

        # 2. Main Splitter (Left: Media + Timeline, Right: Feed + Summary)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {self.config.color_border};
                width: 3px;
            }}
        """)

        # Left Container (approx 65% width)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self.video_view = VideoView(self)
        left_layout.addWidget(self.video_view, stretch=1)

        self.playback_bar = PlaybackBar(self)
        left_layout.addWidget(self.playback_bar)

        self.timeline = ChunkTimeline(self)
        left_layout.addWidget(self.timeline)

        splitter.addWidget(left_container)

        # Right Container (approx 35% width)
        right_container = QWidget()
        right_container.setStyleSheet(f"background-color: {self.config.color_surface};")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.summary_panel = SummaryPanel(self)
        right_layout.addWidget(self.summary_panel)

        self.event_feed = EventFeed(self)
        right_layout.addWidget(self.event_feed, stretch=1)

        splitter.addWidget(right_container)

        # Set 65% / 35% initial width distribution
        splitter.setSizes([832, 448])
        root_layout.addWidget(splitter, stretch=1)

        # 3. Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {self.config.color_surface};
                color: {self.config.color_text_muted};
                border-top: 1px solid {self.config.color_border};
                font-size: 11px;
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a video or image file to begin.")

    def _init_shortcuts(self):
        # Space: Play / Pause
        self.sc_play = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.sc_play.activated.connect(self._toggle_playback)

        # Left Arrow: Seek to previous chunk
        self.sc_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.sc_prev.activated.connect(self._seek_prev_chunk)

        # Right Arrow: Seek to next chunk
        self.sc_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.sc_next.activated.connect(self._seek_next_chunk)

    def _connect_signals(self):
        # Settings Bar
        self.settings_bar.open_clicked.connect(self._on_open_media)
        self.settings_bar.process_clicked.connect(self._start_processing)
        self.settings_bar.cancel_clicked.connect(self._cancel_processing)
        self.settings_bar.export_json_clicked.connect(self._export_json)
        self.settings_bar.export_csv_clicked.connect(self._export_csv)

        # Video View Drag and Drop
        self.video_view.file_dropped.connect(self.load_media)

        # Playback Controls
        self.playback_bar.play_toggled.connect(self._on_play_toggled)
        self.playback_bar.seek_requested.connect(self._on_seek_requested)
        self.playback_bar.speed_changed.connect(self.player_worker.set_speed)

        # Timeline Controls
        self.timeline.seek_requested.connect(self._on_seek_requested)
        self.timeline.chunk_clicked.connect(self._on_seek_requested)

        # Feed Controls
        self.event_feed.chunk_selected.connect(self._on_seek_requested)
        self.event_feed.verdict_updated.connect(self._on_verdict_updated)

        # Player Worker signals
        self.player_worker.frame_ready.connect(self._on_frame_ready)
        self.player_worker.media_loaded.connect(self._on_media_loaded)
        self.player_worker.playback_finished.connect(self._on_playback_finished)
        self.player_worker.error_occurred.connect(self._on_player_error)

    def _on_open_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video or Image File", "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.png *.jpeg);;All Files (*)"
        )
        if file_path:
            self.load_media(file_path)

    def load_media(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"The file '{path}' does not exist.")
            return

        self.session.clear()
        self.session.media_path = path
        self.event_feed.clear()
        self.summary_panel.update_metrics(self.session)
        
        ok = self.player_worker.load_media(path)
        if ok:
            self.current_state = "LOADED"
            self.settings_bar.set_state_loaded()
            self.video_view.set_empty_state(False)
            stem = os.path.basename(path)
            self.status_bar.showMessage(f"Loaded: {stem}. Click [Process] to run VLM analysis.")

    def _on_media_loaded(self, duration_ms: int, fps: float, width: int, height: int):
        self.session.duration_ms = duration_ms
        self.session.fps = fps
        self.playback_bar.set_duration(duration_ms)
        self.timeline.set_duration(duration_ms)

    def _start_processing(self):
        if not self.session.media_path:
            return

        # If already ready, confirm reprocess
        if self.current_state == "READY" and len(self.session.events) > 0:
            res = QMessageBox.question(
                self, "Reprocess Media?",
                "Reprocessing will clear existing results. Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        self.current_state = "PROCESSING"
        self.player_worker.pause()
        self.playback_bar.set_playing_state(False)

        self.session.clear()
        self.event_feed.clear()
        self.summary_panel.update_metrics(self.session)

        chunk_s = self.settings_bar.spin_chunk.value()
        overlap_s = self.settings_bar.spin_overlap.value()
        vlm_fps = float(self.settings_bar.spin_fps.value())

        self.settings_bar.set_state_processing()
        self.status_bar.showMessage("Processing video chunks with VLM...")

        self.batch_worker = BatchWorker(
            media_path=self.session.media_path,
            adapter=self.adapter,
            chunk_s=chunk_s,
            overlap_s=overlap_s,
            vlm_fps=vlm_fps,
            parent=self,
        )
        self.batch_worker.event_ready.connect(self._on_chunk_event_ready)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.canceled.connect(self._on_batch_canceled)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        self.batch_worker.start()

    def _cancel_processing(self):
        if self.batch_worker and self.batch_worker.isRunning():
            self.status_bar.showMessage("Cancelling processing... Preserving completed chunks.")
            self.batch_worker.cancel()

    def _on_chunk_event_ready(self, event: VLMEvent):
        self.session.add_event(event)
        self.event_feed.add_event(event)
        self.timeline.set_events(self.session.events, total_chunks=self.total_planned_chunks)
        self.summary_panel.update_metrics(self.session)

    def _on_batch_progress(self, completed: int, total: int, eta_s: float):
        self.total_planned_chunks = total
        self.settings_bar.update_progress(completed, total, eta_s)

    def _on_batch_finished(self):
        self.current_state = "READY"
        self.settings_bar.set_state_ready(has_results=len(self.session.events) > 0)
        self.timeline.set_events(self.session.events, total_chunks=self.total_planned_chunks)
        self.status_bar.showMessage(f"Processing complete: {len(self.session.events)} chunks processed.")
        # Seek to start
        self._on_seek_requested(0)

    def _on_batch_canceled(self):
        self.current_state = "READY"
        self.settings_bar.set_state_ready(has_results=len(self.session.events) > 0)
        self.timeline.set_events(self.session.events, total_chunks=self.total_planned_chunks)
        self.status_bar.showMessage(f"Processing canceled: {len(self.session.events)} chunks preserved.")

    def _on_batch_error(self, err_msg: str):
        self.current_state = "LOADED"
        self.settings_bar.set_state_loaded()
        QMessageBox.critical(self, "Processing Error", f"Error during VLM batch processing:\n{err_msg}")
        self.status_bar.showMessage("Processing halted due to error.")

    def _toggle_playback(self):
        if self.current_state != "READY":
            return
        is_playing = not self.player_worker.is_playing
        self._on_play_toggled(is_playing)

    def _on_play_toggled(self, play: bool):
        if self.current_state != "READY":
            self.playback_bar.set_playing_state(False)
            return
        if play:
            self.player_worker.play()
            self.playback_bar.set_playing_state(True)
        else:
            self.player_worker.pause()
            self.playback_bar.set_playing_state(False)

    def _on_seek_requested(self, ts_ms: int):
        self.player_worker.pause()
        self.playback_bar.set_playing_state(False)
        self.player_worker.seek(ts_ms)
        self._sync_active_chunk(ts_ms)

    def _on_frame_ready(self, qimage, ts_ms: int):
        self.video_view.set_frame(qimage)
        self.playback_bar.update_timestamp(ts_ms)
        self._sync_active_chunk(ts_ms)

    def _sync_active_chunk(self, ts_ms: int):
        active_event = self.session.get_active_event_at_ts(ts_ms)
        active_chunk_id = active_event.chunk_id if active_event else None
        
        self.video_view.set_active_event(active_event, gt_enabled=self.session.gt_enabled)
        self.timeline.set_playhead(ts_ms)
        self.timeline.set_active_chunk(active_chunk_id)
        self.event_feed.set_playhead(ts_ms, active_chunk_id)

    def _seek_prev_chunk(self):
        if not self.session.events:
            return
        cur_ts = self.player_worker.current_ts_ms
        prev_events = [e for e in self.session.events if e.start_ms < cur_ts - 100]
        if prev_events:
            target = prev_events[-1].start_ms
        else:
            target = 0
        self._on_seek_requested(target)

    def _seek_next_chunk(self):
        if not self.session.events:
            return
        cur_ts = self.player_worker.current_ts_ms
        next_events = [e for e in self.session.events if e.start_ms > cur_ts + 100]
        if next_events:
            target = next_events[0].start_ms
            self._on_seek_requested(target)

    def _on_playback_finished(self):
        self.playback_bar.set_playing_state(False)

    def _on_player_error(self, msg: str):
        QMessageBox.warning(self, "Player Error", msg)

    def _on_verdict_updated(self, event_id: int, verdict: str, note: str):
        ev = self.session.get_event_by_id(event_id)
        if ev:
            ev.verdict = verdict
            ev.note = note
            self.status_bar.showMessage(f"Updated verdict for Chunk #{ev.chunk_id}: {verdict}")

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Results as JSON", "vlm_results.json", "JSON (*.json)")
        if path:
            SessionExporter.export_json(path, self.session, self.config, model_label=self.model_label)
            QMessageBox.information(self, "Export Successful", f"Results successfully exported to:\n{path}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Results as CSV", "vlm_results.csv", "CSV (*.csv)")
        if path:
            SessionExporter.export_csv(path, self.session)
            QMessageBox.information(self, "Export Successful", f"Results successfully exported to:\n{path}")

    def closeEvent(self, event):
        if self.batch_worker and self.batch_worker.isRunning():
            res = QMessageBox.question(
                self, "Analysis In Progress",
                "VLM analysis is currently running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if res != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.batch_worker.cancel()
            self.batch_worker.wait(1000)

        self.player_worker.stop()
        self.adapter.shutdown()
        event.accept()
