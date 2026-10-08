from __future__ import annotations

import os
from datetime import datetime
from time import perf_counter

import cv2
from PySide6.QtCore import Signal

from app.core.chunker import compute_chunk_windows
from app.core.frame_sampler import (
    read_frames_at,
    sample_timestamps,
    target_frame_count,
)
from app.core.gate_log import GateLog, make_log_path
from app.core.jev_adapter import evaluate_tick
from app.core.jev_gate import (
    GateConfig,
    GateRouter,
    TIER_FULL,
    assign_ticks_to_chunks,
    compute_tick_times,
    gate_record_to_dict,
    tick_to_dict,
    validate_against_vlm_fps,
)
from app.core.types import GateRecord, TickRecord, VideoChunk
from app.workers.batch_worker import BatchWorker


class GatedBatchWorker(BatchWorker):
    """
    JEV-gated replacement for BatchWorker (Design Doc SS7.1).

    For each video chunk the worker:
      1. Runs JEV hazard ticks to score the window.
      2. Routes the chunk to FULL or REDUCED frame rate.
      3. Samples frames at the routed fps.
      4. Calls the VLM adapter.

    For image input the base class run() is called unchanged.

    Extra signals (never emitted for image input):
      gate_ready    - GateRecord, emitted before event_ready per chunk
      gate_run_info - dict with run metadata, emitted once at start
      gate_status   - str status messages for the status bar
    """

    gate_ready    = Signal(object)
    gate_run_info = Signal(object)
    gate_status   = Signal(str)

    def __init__(
        self,
        media_path: str,
        adapter,
        chunk_s: float = 5.0,
        overlap_s: float = 1.0,
        vlm_fps: float = 4.0,
        parent=None,
        *,
        decider,
        gate_config: GateConfig,
        log_dir: str,
    ) -> None:
        super().__init__(
            media_path=media_path,
            adapter=adapter,
            chunk_s=chunk_s,
            overlap_s=overlap_s,
            vlm_fps=vlm_fps,
            parent=parent,
        )
        self._decider = decider
        self._gate_cfg = gate_config
        self._log_dir = log_dir
        self._is_hermes = hasattr(self.adapter.vlm, "start_session")

    # ------------------------------------------------------------------
    # Main run loop  (Design Doc SS7.1)
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901
        cfg = self._gate_cfg

        # Step 1: image input -> delegate to baseline, no gate signals/log
        if cv2.imread(self.media_path) is not None:
            super().run()
            return

        # Step 2: open video
        cap = cv2.VideoCapture(self.media_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"Failed to open video: {self.media_path}")
            return

        # Step 3: validate reduced fps vs vlm_fps
        try:
            validate_against_vlm_fps(cfg, self.vlm_fps)
        except ValueError as exc:
            cap.release()
            self.error_occurred.emit(str(exc))
            return

        # Step 4: geometry
        source_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_ms = int(round(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps * 1000.0
        ))
        windows = compute_chunk_windows(
            duration_ms=duration_ms,
            chunk_s=self.chunk_s,
            overlap_s=self.overlap_s,
            include_partial=True,
        )
        if not windows:
            cap.release()
            self.finished.emit()
            return

        # Step 5: tick schedule
        tick_times = compute_tick_times(duration_ms, cfg.window_ms, cfg.stride_ms)
        assigned   = assign_ticks_to_chunks(tick_times, windows)
        total      = len(windows)
        n_ticks    = len(tick_times)

        # Step 6: open log, write run_start, emit gate_run_info
        decider_label = getattr(self._decider, "label", type(self._decider).__name__)
        log_path = make_log_path(self._log_dir, self.media_path, datetime.now())
        log = GateLog(log_path)
        log.write({
            "type":        "run_start",
            "time":        datetime.now().astimezone().isoformat(),
            "media":       self.media_path,
            "duration_ms": duration_ms,
            "chunk_s":     self.chunk_s,
            "overlap_s":   self.overlap_s,
            "vlm_fps":     self.vlm_fps,
            "n_chunks":    total,
            "n_ticks":     n_ticks,
            "gate_config": cfg.to_dict(),
            "decider":     decider_label,
        })
        self.gate_run_info.emit({
            "media":       self.media_path,
            "duration_ms": duration_ms,
            "n_chunks":    total,
            "n_ticks":     n_ticks,
            "gate_config": cfg.to_dict(),
            "decider":     decider_label,
            "log_path":    log.path,
        })

        # Step 7: start HERMES session (if applicable)
        if self._is_hermes:
            self.adapter.vlm.start_session()

        _log_warn_done = False
        completed = 0

        try:
            router     = GateRouter(cfg)
            start_perf = perf_counter()

            for idx, (w_start, w_end) in enumerate(windows):

                # 8a: cancel check before ticks
                if self._is_canceled:
                    break

                # 8b: run JEV ticks for this chunk
                collected: list[TickRecord] = []
                for tick_id in assigned[idx]:
                    if self._is_canceled:
                        break
                    t_ms = tick_times[tick_id]
                    self.gate_status.emit(
                        f"JEV tick {tick_id + 1}/{n_ticks} \u00b7 chunk {idx + 1}/{total}"
                    )
                    tick_rec = evaluate_tick(
                        self._decider, cap, tick_id, t_ms, cfg
                    )
                    ok = log.write(
                        {"type": "tick", "chunk_id": idx, **tick_to_dict(tick_rec)}
                    )
                    if not ok and not _log_warn_done:
                        self.gate_status.emit(f"gate log write failed: {log.error}")
                        _log_warn_done = True
                    collected.append(tick_rec)

                # 8c: cancel after ticks -> chunk NOT forwarded to VLM
                if self._is_canceled:
                    break

                # 8d: route decision
                decision = router.decide(collected)
                tier   = decision.tier
                reason = decision.reason

                # 8e: sample frames at tier fps
                fps_used = self.vlm_fps if tier == TIER_FULL else cfg.route_reduced_fps
                ts_list  = sample_timestamps(w_start, w_end, fps_used)
                frames, timestamps = read_frames_at(cap, ts_list, rgb=True)
                chunk = VideoChunk(
                    chunk_id=idx,
                    source=self.media_path,
                    start_ms=w_start,
                    end_ms=w_end,
                    fps=fps_used,
                    frames=frames,
                    timestamps_ms=timestamps,
                )

                # 8f: status update, then VLM call
                self.gate_status.emit(
                    f"VLM chunk {idx + 1}/{total} \u00b7 {tier} @ {fps_used:g} fps ({reason})"
                )
                event       = self.adapter.process_chunk(chunk, event_id=idx + 1)
                wall_done_s = perf_counter() - start_perf

                # 8g: build GateRecord
                # top_tick_id = ok tick with highest pmax; ties -> earliest tick_id
                ok_ticks = [
                    t for t in collected
                    if t.status == "ok" and t.probs is not None
                ]
                if ok_ticks:
                    top = max(
                        ok_ticks,
                        key=lambda t: (max(t.probs.values()), -t.tick_id),
                    )
                    top_tick_id: int | None = top.tick_id
                else:
                    top_tick_id = None

                # Retain thumbs only on the top tick
                ticks_stored: list[TickRecord] = []
                for t in collected:
                    if t.tick_id == top_tick_id:
                        ticks_stored.append(t)
                    else:
                        ticks_stored.append(TickRecord(
                            tick_id=t.tick_id,
                            t_ms=t.t_ms,
                            window_start_ms=t.window_start_ms,
                            num_frames=t.num_frames,
                            status=t.status,
                            probs=t.probs,
                            error=t.error,
                            jev_time_s=t.jev_time_s,
                            thumbs=[],
                        ))

                frames_full_equiv = target_frame_count(w_start, w_end, self.vlm_fps)
                gate_rec = GateRecord(
                    chunk_id=idx,
                    start_ms=w_start,
                    end_ms=w_end,
                    tier=tier,
                    reason=reason,
                    vec=decision.vec,
                    pmax=decision.pmax,
                    rising_hazard=decision.rising_hazard,
                    tick_ids=[t.tick_id for t in collected],
                    ticks=ticks_stored,
                    top_tick_id=top_tick_id,
                    num_ticks_ok=sum(1 for t in collected if t.status == "ok"),
                    num_ticks_error=sum(1 for t in collected if t.status == "error"),
                    sample_fps=fps_used,
                    frames_sent=len(frames),
                    frames_full_equiv=frames_full_equiv,
                    jev_time_s=sum(t.jev_time_s for t in collected),
                    vlm_time_s=event.latency_s,
                    vlm_status=event.status,
                    vlm_score=event.score,
                    wall_done_s=wall_done_s,
                    hold_remaining_after=decision.hold_remaining_after,
                    ewma=decision.ewma,
                )

                ok = log.write(
                    {"type": "chunk",
                     **gate_record_to_dict(gate_rec, include_ticks=False)}
                )
                if not ok and not _log_warn_done:
                    self.gate_status.emit(f"gate log write failed: {log.error}")
                    _log_warn_done = True

                # Emit gate_ready THEN event_ready (SS7.1 §8g ordering)
                self.gate_ready.emit(gate_rec)
                self.event_ready.emit(event)

                completed = idx + 1
                elapsed   = perf_counter() - start_perf
                eta_s     = (elapsed / completed) * (total - completed)
                self.progress.emit(completed, total, eta_s)

            # Step 9: teardown after loop exits (finished or canceled)
            cap.release()
            if self._is_hermes:
                try:
                    self.adapter.vlm.end_session()
                except Exception:
                    pass

            outcome = "canceled" if self._is_canceled else "finished"
            log.write({
                "type":        "run_end",
                "outcome":     outcome,
                "chunks_done": completed,
                "wall_s":      perf_counter() - start_perf,
                "error":       None,
            })
            log.close()

            if self._is_canceled:
                self.canceled.emit()
            else:
                self.finished.emit()

        except Exception as exc:
            # Step 10: exception handler
            cap.release()
            if self._is_hermes:
                try:
                    self.adapter.vlm.end_session()
                except Exception:
                    pass
            log.write({
                "type":        "run_end",
                "outcome":     "error",
                "chunks_done": completed,
                "wall_s":      0.0,
                "error":       str(exc),
            })
            log.close()
            self.error_occurred.emit(str(exc))
