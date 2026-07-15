from dataclasses import dataclass
from threading import Lock
import time

from withnuri.pipeline.display_stabilizer import DisplayStabilizationMetrics

@dataclass(frozen=True)
class PipelineDiagnostics:
    """Aggregate local pipeline timing and tracking continuity measurements."""

    elapsed_seconds: float
    decoded_frame_count: int
    dropped_frame_count: int
    skipped_inference_frame_count: int
    processed_frame_count: int
    detected_frame_count: int
    track_loss_events: int
    track_recovery_events: int
    track_id_change_events: int
    average_queue_delay_ms: float
    max_queue_delay_ms: float
    average_inference_ms: float
    max_inference_ms: float
    average_render_ms: float
    max_render_ms: float
    average_present_ms: float
    max_present_ms: float
    display_metrics: DisplayStabilizationMetrics | None = None

    @property
    def decoded_fps(self) -> float:
        return _per_second(self.decoded_frame_count, self.elapsed_seconds)

    @property
    def processed_fps(self) -> float:
        return _per_second(self.processed_frame_count, self.elapsed_seconds)


class PipelineDiagnosticsCollector:
    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._lock = Lock()
        self._decoded_frame_count = 0
        self._skipped_inference_frame_count = 0
        self._processed_frame_count = 0
        self._detected_frame_count = 0
        self._track_loss_events = 0
        self._track_recovery_events = 0
        self._track_id_change_events = 0
        self._previous_detected = False
        self._has_seen_detection = False
        self._previous_track_ids: frozenset[int] = frozenset()
        self._queue_delay_total_seconds = 0.0
        self._queue_delay_max_seconds = 0.0
        self._inference_total_seconds = 0.0
        self._inference_max_seconds = 0.0
        self._render_total_seconds = 0.0
        self._render_max_seconds = 0.0
        self._present_total_seconds = 0.0
        self._present_max_seconds = 0.0

    def record_decoded_frame(self) -> None:
        with self._lock:
            self._decoded_frame_count += 1

    def record_skipped_inference_frame(self) -> None:
        with self._lock:
            self._skipped_inference_frame_count += 1

    def record_processed_frame(
        self,
        *,
        queue_delay_seconds: float,
        inference_seconds: float,
        render_seconds: float,
        present_seconds: float,
        detected: bool,
        track_ids: set[int],
    ) -> None:
        current_track_ids = frozenset(track_ids)
        with self._lock:
            self._processed_frame_count += 1
            self._queue_delay_total_seconds, self._queue_delay_max_seconds = _add_sample(
                self._queue_delay_total_seconds,
                self._queue_delay_max_seconds,
                queue_delay_seconds,
            )
            self._inference_total_seconds, self._inference_max_seconds = _add_sample(
                self._inference_total_seconds,
                self._inference_max_seconds,
                inference_seconds,
            )
            self._render_total_seconds, self._render_max_seconds = _add_sample(
                self._render_total_seconds,
                self._render_max_seconds,
                render_seconds,
            )
            self._present_total_seconds, self._present_max_seconds = _add_sample(
                self._present_total_seconds,
                self._present_max_seconds,
                present_seconds,
            )
            if detected:
                self._detected_frame_count += 1
            if self._previous_detected and not detected:
                self._track_loss_events += 1
            elif not self._previous_detected and detected and self._has_seen_detection:
                self._track_recovery_events += 1
            if (
                detected
                and self._previous_detected
                and current_track_ids
                and self._previous_track_ids
                and current_track_ids != self._previous_track_ids
            ):
                self._track_id_change_events += 1
            self._previous_detected = detected
            self._has_seen_detection = self._has_seen_detection or detected
            self._previous_track_ids = current_track_ids

    def snapshot(
        self,
        *,
        dropped_frame_count: int,
        display_metrics: DisplayStabilizationMetrics | None = None,
    ) -> PipelineDiagnostics:
        with self._lock:
            elapsed_seconds = max(0.0, self._clock() - self._started_at)
            return PipelineDiagnostics(
                elapsed_seconds=elapsed_seconds,
                decoded_frame_count=self._decoded_frame_count,
                dropped_frame_count=dropped_frame_count,
                skipped_inference_frame_count=self._skipped_inference_frame_count,
                processed_frame_count=self._processed_frame_count,
                detected_frame_count=self._detected_frame_count,
                track_loss_events=self._track_loss_events,
                track_recovery_events=self._track_recovery_events,
                track_id_change_events=self._track_id_change_events,
                average_queue_delay_ms=_average_ms(
                    self._queue_delay_total_seconds, self._processed_frame_count
                ),
                max_queue_delay_ms=self._queue_delay_max_seconds * 1_000,
                average_inference_ms=_average_ms(
                    self._inference_total_seconds, self._processed_frame_count
                ),
                max_inference_ms=self._inference_max_seconds * 1_000,
                average_render_ms=_average_ms(
                    self._render_total_seconds, self._processed_frame_count
                ),
                max_render_ms=self._render_max_seconds * 1_000,
                average_present_ms=_average_ms(
                    self._present_total_seconds, self._processed_frame_count
                ),
                max_present_ms=self._present_max_seconds * 1_000,
                display_metrics=display_metrics,
            )


def _per_second(count: int, elapsed_seconds: float) -> float:
    return count / elapsed_seconds if elapsed_seconds > 0 else 0.0


def _add_sample(total: float, maximum: float, value: float) -> tuple[float, float]:
    value = max(0.0, value)
    return total + value, max(maximum, value)


def _average_ms(total_seconds: float, sample_count: int) -> float:
    return total_seconds * 1_000 / sample_count if sample_count else 0.0
