import threading
import time

from withnuri.pipeline.debug_overlay import render_tracking_debug_overlay
from withnuri.pipeline.diagnostics import PipelineDiagnosticsCollector
from withnuri.pipeline.frames import ProcessedFrame
from withnuri.pipeline.queue import LatestFrameQueue
from withnuri.ui.windowing import (
    PreviewResult,
    PreviewWindowUnavailable,
    resize_processed_frame,
)


class QtDebugWindowUnavailable(PreviewWindowUnavailable):
    """Raised when PySide6 is unavailable for debug windows."""


class QtDebugWindow:
    def __init__(
        self,
        *,
        title: str = "WithNuri Debug Overlay",
        always_on_top: bool = False,
        display_width: int | None = None,
        display_height: int | None = None,
        qt_modules: dict | None = None,
    ):
        modules = qt_modules or _load_qt_modules()
        self._qapplication = modules["QApplication"]
        self._qwidget = modules["QWidget"]
        self._qlabel = modules["QLabel"]
        self._qvboxlayout = modules["QVBoxLayout"]
        self._qimage = modules["QImage"]
        self._qpixmap = modules["QPixmap"]
        self._qt = modules["Qt"]
        self._display_width = display_width
        self._display_height = display_height
        self._closed = False
        self._last_image = None

        self._app = self._qapplication.instance() or self._qapplication([])
        self._widget = self._qwidget()
        self._widget.setWindowTitle(title)
        if always_on_top:
            self._widget.setWindowFlags(self._qt.WindowStaysOnTopHint)
        self._label = self._qlabel()
        layout = self._qvboxlayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._widget.setLayout(layout)
        self._widget.show()

    def show_frame(self, frame: ProcessedFrame) -> None:
        if self._closed:
            return
        if self._display_width is not None and self._display_height is not None:
            frame = resize_processed_frame(
                frame, width=self._display_width, height=self._display_height
            )
        image = self._qimage(
            frame.data,
            frame.width,
            frame.height,
            self._qimage.Format_RGBA8888,
        ).copy()
        self._last_image = image
        self._label.setPixmap(self._qpixmap.fromImage(image))
        self._app.processEvents()

    def should_close(self) -> bool:
        # The widget stops being visible when the user closes it through the
        # window manager, even though our own close() was never called.
        if self._closed:
            return True
        return not self._widget.isVisible()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._widget.close()
        self._app.processEvents()


def run_tracking_debug_preview(
    decoder,
    *,
    tracker,
    window,
    max_frames: int | None = None,
    sleep=time.sleep,
    idle_seconds: float = 0.005,
    collect_diagnostics: bool = False,
    clock=time.monotonic,
) -> PreviewResult:
    # Decode runs on its own thread and only ever hands the consumer the most
    # recent frame; if tracking/inference is slower than decode, intermediate
    # frames are dropped instead of piling up latency. Qt requires the consumer
    # (window.show_frame) to stay on the main thread, so decode is the producer.
    queue: LatestFrameQueue = LatestFrameQueue()
    stop = threading.Event()
    producer_error: list[BaseException] = []
    frame_iter = decoder.iter_frames()
    diagnostics = (
        PipelineDiagnosticsCollector(clock=clock) if collect_diagnostics else None
    )

    def produce() -> None:
        try:
            for raw_frame in frame_iter:
                if stop.is_set():
                    break
                if diagnostics is not None:
                    diagnostics.record_decoded_frame()
                queue.put(raw_frame)
        except BaseException as exc:  # surfaced to the consumer after join
            producer_error.append(exc)
        finally:
            stop.set()
            # Safe to close here: the generator is owned by this thread, so its
            # cleanup (terminating ffmpeg) runs without cross-thread races.
            frame_iter.close()

    thread = threading.Thread(target=produce, name="withnuri-decode", daemon=True)

    frames_rendered = 0
    visible_frame_count = 0
    last_frame_visible = False
    interrupted = False

    thread.start()
    try:
        try:
            while True:
                raw_frame = queue.latest()
                if raw_frame is None:
                    if stop.is_set():
                        break
                    sleep(idle_seconds)
                    continue
                processing_started_at = clock()
                tracking = tracker.track(raw_frame)
                inference_seconds = clock() - processing_started_at
                debug_frame = render_tracking_debug_overlay(raw_frame, tracking)
                render_seconds = clock() - processing_started_at - inference_seconds
                present_started_at = clock()
                window.show_frame(debug_frame)
                present_seconds = clock() - present_started_at
                frames_rendered += 1
                last_frame_visible = any(track.detected for track in tracking.tracks)
                if last_frame_visible:
                    visible_frame_count += 1
                if diagnostics is not None:
                    diagnostics.record_processed_frame(
                        queue_delay_seconds=processing_started_at
                        - raw_frame.timestamp_seconds,
                        inference_seconds=inference_seconds,
                        render_seconds=render_seconds,
                        present_seconds=present_seconds,
                        detected=last_frame_visible,
                        track_ids={
                            track.track_id
                            for track in tracking.tracks
                            if track.detected and track.track_id is not None
                        },
                    )
                if window.should_close():
                    break
                if max_frames is not None and frames_rendered >= max_frames:
                    break
        except KeyboardInterrupt:
            interrupted = True
    finally:
        stop.set()
        thread.join(timeout=5)
        window.close()

    if producer_error:
        error = producer_error[0]
        if isinstance(error, KeyboardInterrupt):
            interrupted = True
        elif not interrupted:
            raise error

    return PreviewResult(
        frames_rendered=frames_rendered,
        interrupted=interrupted,
        visible_frame_count=visible_frame_count,
        last_frame_visible=last_frame_visible,
        diagnostics=(
            diagnostics.snapshot(dropped_frame_count=queue.dropped_frame_count)
            if diagnostics is not None
            else None
        ),
    )


def _load_qt_modules() -> dict:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    except ImportError as exc:
        raise QtDebugWindowUnavailable("PySide6 is required for debug windows") from exc
    return {
        "QApplication": QApplication,
        "QWidget": QWidget,
        "QLabel": QLabel,
        "QVBoxLayout": QVBoxLayout,
        "QImage": QImage,
        "QPixmap": QPixmap,
        "Qt": Qt,
    }
