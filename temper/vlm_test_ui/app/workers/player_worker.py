from __future__ import annotations
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage


def cv2_frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Converts a BGR OpenCV numpy frame to a PySide6 QImage in RGB888."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    # Make copy to ensure buffer memory safety across Qt threads
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class PlayerWorker(QThread):
    """
    Dedicated QThread for real-time video playback using OpenCV.
    Stateless and decoupled from VLM inference. Emits QImage frames with current timestamp.
    """
    frame_ready = Signal(QImage, int)            # QImage, timestamp_ms
    media_loaded = Signal(int, float, int, int)   # duration_ms, fps, width, height
    playback_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.media_path: str = ""
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.duration_ms: int = 0
        self.width: int = 0
        self.height: int = 0
        self.is_image: bool = False
        
        self.current_ts_ms: int = 0
        self.speed: float = 1.0
        self.is_playing: bool = False
        self._is_running: bool = True
        
        self._seek_requested: bool = False
        self._target_seek_ms: int = 0
        
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def load_media(self, path: str) -> bool:
        self._mutex.lock()
        try:
            self.is_playing = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.media_path = path
            # Check if image
            test_img = cv2.imread(path)
            if test_img is not None:
                self.is_image = True
                h, w = test_img.shape[:2]
                self.width, self.height = w, h
                self.duration_ms = 0
                self.fps = 1.0
                self.current_ts_ms = 0
                qimg = cv2_frame_to_qimage(test_img)
                self.media_loaded.emit(0, 1.0, w, h)
                self.frame_ready.emit(qimg, 0)
                return True

            self.is_image = False
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Could not open video file: {path}")
                return False

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration_ms = int(round((frame_count / self.fps) * 1000.0))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Emit media metadata
            self.media_loaded.emit(self.duration_ms, self.fps, self.width, self.height)
            
            # Read first frame
            ret, frame = self.cap.read()
            if ret:
                self.current_ts_ms = 0
                self.frame_ready.emit(cv2_frame_to_qimage(frame), 0)
            return True
        finally:
            self._mutex.unlock()

    def play(self):
        self._mutex.lock()
        if not self.is_playing and not self.is_image and self.cap is not None:
            # If at the end, restart from beginning
            if self.current_ts_ms >= self.duration_ms:
                self._seek_requested = True
                self._target_seek_ms = 0
            self.is_playing = True
            self._condition.wakeAll()
        self._mutex.unlock()

    def pause(self):
        self._mutex.lock()
        self.is_playing = False
        self._mutex.unlock()

    def set_speed(self, speed: float):
        self._mutex.lock()
        self.speed = max(0.25, min(4.0, speed))
        self._mutex.unlock()

    def seek(self, ts_ms: int):
        self._mutex.lock()
        if self.is_image:
            self._mutex.unlock()
            return
            
        self._target_seek_ms = max(0, min(self.duration_ms, ts_ms))
        self._seek_requested = True
        self._condition.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._is_running = False
        self.is_playing = False
        self._condition.wakeAll()
        self._mutex.unlock()
        self.wait(1000)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        while True:
            self._mutex.lock()
            if not self._is_running:
                self._mutex.unlock()
                break

            # Handle seeking
            if self._seek_requested and self.cap is not None:
                self._seek_requested = False
                target_ms = self._target_seek_ms
                self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                ret, frame = self.cap.read()
                if ret:
                    self.current_ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                    qimg = cv2_frame_to_qimage(frame)
                    self.frame_ready.emit(qimg, self.current_ts_ms)
                self._mutex.unlock()
                continue

            if not self.is_playing or self.cap is None:
                self._condition.wait(self._mutex, 100)
                self._mutex.unlock()
                continue

            # Playback step
            speed = self.speed
            fps = self.fps
            self._mutex.unlock()

            frame_interval_s = (1.0 / (fps * speed)) if (fps * speed) > 0 else 0.033
            start_tick = time.perf_counter()

            self._mutex.lock()
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    self.current_ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                    qimg = cv2_frame_to_qimage(frame)
                    self.frame_ready.emit(qimg, self.current_ts_ms)
                else:
                    # End of stream
                    self.is_playing = False
                    self.playback_finished.emit()
            self._mutex.unlock()

            elapsed = time.perf_counter() - start_tick
            sleep_time = max(0.001, frame_interval_s - elapsed)
            time.sleep(sleep_time)
