import threading
import time

from withnuri.pipeline.frames import ProcessedFrame
from withnuri.pipeline.matte import render_pet_matte, scale_processed_alpha
from withnuri.pipeline.queue import LatestFrameQueue
from withnuri.ui.windowing import (
    PreviewResult,
    PreviewWindowUnavailable,
    resize_processed_frame,
)


class OverlayWindowUnavailable(PreviewWindowUnavailable):
    """Raised when PySide6 is unavailable for the transparent overlay."""


class OverlayWindow:
    def __init__(
        self,
        *,
        panel_width: int = 480,
        panel_height: int = 270,
        margin: int = 24,
        qt_modules: dict | None = None,
        screen_geometry: tuple[int, int, int, int] | None = None,
    ):
        modules = qt_modules or _load_overlay_qt_modules()
        self._qapplication = modules["QApplication"]
        self._qimage = modules["QImage"]
        self._qpixmap = modules["QPixmap"]
        self._qt = modules["Qt"]
        self._panel_width = panel_width
        self._panel_height = panel_height
        self._margin = margin
        self._closed = False
        self._last_image = None
        self._click_through = True
        self._drag_origin = None
        self._drag_window_start = None

        self._app = self._qapplication.instance() or self._qapplication([])

        widget_class = modules["QWidget"]

        class _DraggableWidget(widget_class):
            def __init__(self_w, window):
                super().__init__()
                self_w._window = window

            def mousePressEvent(self_w, event):
                pos = event.globalPosition().toPoint()
                self_w._window.begin_drag(pos.x(), pos.y())

            def mouseMoveEvent(self_w, event):
                pos = event.globalPosition().toPoint()
                self_w._window.drag_to(pos.x(), pos.y())

        self._widget = _DraggableWidget(self)
        self._widget.setWindowTitle("WithNuri")
        self._widget.setWindowFlags(
            self._qt.FramelessWindowHint
            | self._qt.WindowStaysOnTopHint
            | self._qt.Tool
            | self._qt.WindowTransparentForInput
        )
        self._widget.setAttribute(self._qt.WA_TranslucentBackground, True)
        self._widget.setAttribute(self._qt.WA_NoSystemBackground, True)

        self._label = modules["QLabel"]()
        self._label.setStyleSheet("background: transparent;")
        layout = modules["QVBoxLayout"]()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._widget.setLayout(layout)

        self._widget.resize(panel_width, panel_height)
        self._screen_geometry = screen_geometry
        self._position = (0, 0)
        self.move_to_corner()
        self._widget.show()

    @property
    def widget(self):
        return self._widget

    def move_to_corner(self) -> None:
        geometry = self._screen_geometry or self._primary_geometry()
        x, y, width, height = geometry
        self._move(
            x + width - self._panel_width - self._margin,
            y + height - self._panel_height - self._margin,
        )

    def _move(self, x: int, y: int) -> None:
        self._position = (x, y)
        self._widget.move(x, y)

    def show_frame(self, frame: ProcessedFrame) -> None:
        if self._closed:
            return
        frame = resize_processed_frame(
            frame, width=self._panel_width, height=self._panel_height
        )
        image = self._qimage(
            frame.data,
            frame.width,
            frame.height,
            self._qimage.Format_RGBA8888_Premultiplied,
        ).copy()
        self._last_image = image
        self._label.setPixmap(self._qpixmap.fromImage(image))
        self._app.processEvents()

    def should_close(self) -> bool:
        if self._closed:
            return True
        return not self._widget.isVisible()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._widget.close()
        self._app.processEvents()

    def is_click_through(self) -> bool:
        return self._click_through

    def set_click_through(self, enabled: bool) -> None:
        if enabled == self._click_through:
            return
        self._click_through = enabled
        base = (
            self._qt.FramelessWindowHint
            | self._qt.WindowStaysOnTopHint
            | self._qt.Tool
        )
        if enabled:
            base |= self._qt.WindowTransparentForInput
        self._widget.setWindowFlags(base)
        self._widget.show()  # flag changes require re-show

    def begin_drag(self, global_x: int, global_y: int) -> None:
        self._drag_origin = (global_x, global_y)
        self._drag_window_start = self._position

    def drag_to(self, global_x: int, global_y: int) -> None:
        if self._drag_origin is None or self._drag_window_start is None:
            return
        dx = global_x - self._drag_origin[0]
        dy = global_y - self._drag_origin[1]
        start_x, start_y = self._drag_window_start
        self._move(start_x + dx, start_y + dy)

    def _primary_geometry(self) -> tuple[int, int, int, int]:
        screen = self._app.primaryScreen()
        rect = screen.availableGeometry()
        return (rect.x(), rect.y(), rect.width(), rect.height())


class FadeController:
    def __init__(
        self,
        *,
        hold_seconds: float = 0.6,
        fade_seconds: float = 0.5,
        clock=time.monotonic,
    ):
        self._hold_seconds = hold_seconds
        self._fade_seconds = fade_seconds
        self._clock = clock
        self._last_good: ProcessedFrame | None = None
        self._lost_at: float | None = None

    def display_frame(self, fresh: ProcessedFrame, has_pet: bool) -> ProcessedFrame:
        if has_pet:
            self._last_good = fresh
            self._lost_at = None
            return fresh
        if self._last_good is None:
            return fresh
        if self._lost_at is None:
            self._lost_at = self._clock()
        elapsed = self._clock() - self._lost_at
        if elapsed <= self._hold_seconds:
            return self._last_good
        if self._fade_seconds <= 0:
            return scale_processed_alpha(self._last_good, 0.0)
        ramp = (elapsed - self._hold_seconds) / self._fade_seconds
        factor = max(0.0, 1.0 - ramp)
        return scale_processed_alpha(self._last_good, factor)


class _OverlayTray:
    def __init__(self, icon, menu, arrange_action, window, on_quit):
        self._icon = icon
        self._menu = menu
        self._arrange_action = arrange_action
        self._window = window
        self._on_quit = on_quit

    @property
    def icon(self):
        return self._icon

    def toggle_arrange_mode(self) -> None:
        arrange_on = not self._arrange_action.isChecked()
        self._arrange_action.setChecked(arrange_on)
        # arrange mode ON means the window must NOT be click-through
        self._window.set_click_through(not arrange_on)

    def quit(self) -> None:
        self._on_quit()


def build_overlay_tray(window, *, on_quit, qt_modules: dict | None = None):
    modules = qt_modules or _load_tray_qt_modules()
    icon = modules["QSystemTrayIcon"](modules["QIcon"]())
    menu = modules["QMenu"]()

    arrange_action = modules["QAction"]("Arrange mode")
    arrange_action.setCheckable(True)
    arrange_action.setChecked(False)
    quit_action = modules["QAction"]("Quit")

    menu.addAction(arrange_action)
    menu.addSeparator()
    menu.addAction(quit_action)
    icon.setContextMenu(menu)
    icon.setToolTip("WithNuri")
    icon.show()

    tray = _OverlayTray(icon, menu, arrange_action, window, on_quit)
    # QAction.triggered emits a bool (checked state) as a positional argument;
    # the lambdas intentionally swallow it so the no-arg methods work correctly.
    arrange_action.triggered.connect(lambda *_: tray.toggle_arrange_mode())
    quit_action.triggered.connect(lambda *_: tray.quit())
    return tray


def run_pet_overlay(
    decoder,
    *,
    tracker,
    window,
    app_runner,
    renderer=render_pet_matte,
    fade=None,
    feather_radius: float = 1.5,
    max_frames: int | None = None,
    sleep=time.sleep,
    idle_seconds: float = 0.005,
) -> PreviewResult:
    queue: LatestFrameQueue = LatestFrameQueue()
    stop = threading.Event()
    producer_error: list[BaseException] = []
    frame_iter = decoder.iter_frames()

    def produce() -> None:
        try:
            for raw_frame in frame_iter:
                if stop.is_set():
                    break
                queue.put(raw_frame)
        except BaseException as exc:  # surfaced to the consumer after join
            producer_error.append(exc)
        finally:
            stop.set()
            frame_iter.close()

    thread = threading.Thread(target=produce, name="withnuri-overlay-decode", daemon=True)

    state = {"frames_rendered": 0, "visible": 0, "last_visible": False}

    def tick() -> bool:
        raw_frame = queue.latest()
        if raw_frame is None:
            if stop.is_set():
                return False
            sleep(idle_seconds)
            return True
        tracking = tracker.track(raw_frame)
        fresh = renderer(raw_frame, tracking, feather_radius=feather_radius)
        has_pet = bool(tracking.tracks)
        frame_to_show = fade.display_frame(fresh, has_pet) if fade else fresh
        window.show_frame(frame_to_show)
        state["frames_rendered"] += 1
        state["last_visible"] = has_pet
        if has_pet:
            state["visible"] += 1
        if window.should_close():
            return False
        if max_frames is not None and state["frames_rendered"] >= max_frames:
            return False
        return not stop.is_set()

    interrupted = False
    thread.start()
    try:
        app_runner(tick)
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
        frames_rendered=state["frames_rendered"],
        interrupted=interrupted,
        visible_frame_count=state["visible"],
        last_frame_visible=state["last_visible"],
    )


def _load_tray_qt_modules() -> dict:
    try:
        from PySide6.QtGui import QAction, QIcon
        from PySide6.QtWidgets import QMenu, QSystemTrayIcon
    except ImportError as exc:
        raise OverlayWindowUnavailable(
            "PySide6 is required for the overlay system tray"
        ) from exc
    return {
        "QSystemTrayIcon": QSystemTrayIcon,
        "QMenu": QMenu,
        "QIcon": QIcon,
        "QAction": QAction,
    }


def _load_overlay_qt_modules() -> dict:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    except ImportError as exc:
        raise OverlayWindowUnavailable(
            "PySide6 is required for the transparent overlay"
        ) from exc
    return {
        "QApplication": QApplication,
        "QWidget": QWidget,
        "QLabel": QLabel,
        "QVBoxLayout": QVBoxLayout,
        "QImage": QImage,
        "QPixmap": QPixmap,
        "Qt": Qt,
    }
