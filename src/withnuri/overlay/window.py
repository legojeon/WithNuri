import threading
import time
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
import signal

from withnuri.pipeline.frames import ProcessedFrame
from withnuri.pipeline.inference_gate import IdleInferenceGate
from withnuri.pipeline.diagnostics import PipelineDiagnosticsCollector
from withnuri.pipeline.display_stabilizer import PetDisplayStabilizer
from withnuri.pipeline.matte import render_pet_matte, scale_processed_alpha
from withnuri.pipeline.mask_smoothing import TemporalMaskSmoother
from withnuri.pipeline.pet_crop import PetCropController
from withnuri.pipeline.queue import LatestFrameQueue
from withnuri.streaming.decoder import DecodeToolMissing
from withnuri.overlay.placement import OverlayPlacement, OverlayPlacementStore
from withnuri.ui.windowing import (
    PreviewResult,
    PreviewWindowUnavailable,
    resize_processed_frame,
)


class OverlayWindowUnavailable(PreviewWindowUnavailable):
    """Raised when PySide6 is unavailable for the transparent overlay."""


class OverlayInteractionMode(str, Enum):
    """The only two input modes an overlay panel can be in."""

    OVERLAY = "overlay"
    ARRANGE = "arrange"


class _NoopPlacementStore:
    """Avoid host-user setting writes from injected Qt test environments."""

    def load(self) -> None:
        return None

    def save(self, _placement: OverlayPlacement) -> None:
        return None

    def clear(self) -> None:
        return None


class OverlayWindow:
    _ASPECT_RATIO = 16 / 9
    _MIN_PANEL_WIDTH = 96
    _RESIZE_HANDLE_SIZE = 18

    def __init__(
        self,
        *,
        panel_width: int = 480,
        panel_height: int = 270,
        margin: int = 24,
        qt_modules: dict | None = None,
        screen_geometry: tuple[int, int, int, int] | None = None,
        placement_store: OverlayPlacementStore | None = None,
    ):
        modules = qt_modules or _load_overlay_qt_modules()
        self._qapplication = modules["QApplication"]
        self._qimage = modules["QImage"]
        self._qpixmap = modules["QPixmap"]
        self._qpainter = modules.get("QPainter")
        self._qpen = modules.get("QPen")
        self._qcolor = modules.get("QColor")
        self._qt = modules["Qt"]
        self._panel_width = panel_width
        self._panel_height = panel_height
        self._default_panel_size = (panel_width, panel_height)
        self._margin = margin
        self._placement_store = placement_store or (
            OverlayPlacementStore() if qt_modules is None else _NoopPlacementStore()
        )
        self._closed = False
        self._last_image = None
        self._last_frame: ProcessedFrame | None = None
        self._pixmap = None
        self._mode = OverlayInteractionMode.OVERLAY
        self._drag_origin = None
        self._drag_window_start = None
        self._resize_origin = None
        self._resize_size_start = None
        self._pointer_interaction_active = False
        self._panel_resize_handler: Callable[[int, int], None] | None = None

        self._app = self._qapplication.instance() or self._qapplication([])

        widget_class = modules["QWidget"]

        class _DraggableWidget(widget_class):
            def __init__(self_w, window):
                super().__init__()
                self_w._window = window

            def mousePressEvent(self_w, event):
                if not self_w._window.is_arrange_mode():
                    event.ignore()
                    return
                pos = event.globalPosition().toPoint()
                local = event.position().toPoint()
                self_w._window.begin_pointer_interaction(
                    pos.x(), pos.y(), local.x(), local.y()
                )
                grab_mouse = getattr(self_w, "grabMouse", None)
                if callable(grab_mouse):
                    grab_mouse()

            def mouseMoveEvent(self_w, event):
                if not self_w._window.pointer_interaction_active:
                    return
                pos = event.globalPosition().toPoint()
                self_w._window.move_pointer(pos.x(), pos.y())

            def mouseReleaseEvent(self_w, _event):
                self_w._window.end_pointer_interaction()
                release_mouse = getattr(self_w, "releaseMouse", None)
                if callable(release_mouse):
                    release_mouse()

            def resizeEvent(self_w, event):
                try:
                    super().resizeEvent(event)
                except AttributeError:
                    pass

            def paintEvent(self_w, _event):
                self_w._window._paint_overlay(self_w)

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
        # Qt::Tool is implemented as an NSPanel on macOS. Without this
        # attribute macOS hides the panel as soon as another app becomes
        # active, which makes the click-through overlay disappear after a
        # click on the window behind it.
        mac_always_show_tool = getattr(
            self._qt, "WA_MacAlwaysShowToolWindow", None
        )
        if mac_always_show_tool is not None:
            self._widget.setAttribute(mac_always_show_tool, True)

        self._widget.resize(panel_width, panel_height)
        self._screen_geometry = screen_geometry
        self._position = (0, 0)
        self._restore_placement()
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
        self._last_frame = frame
        frame = resize_processed_frame(
            frame, width=self._panel_width, height=self._panel_height
        )
        image = self._qimage(
            frame.data,
            frame.width,
            frame.height,
            self._qimage.Format_RGBA8888_Premultiplied,
        ).copy()
        self._last_image = image  # retains the QImage backing buffer alive for the pixmap
        self._pixmap = self._qpixmap.fromImage(image)
        self._widget.update()

    def should_close(self) -> bool:
        if self._closed:
            return True
        return not self._widget.isVisible()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._save_placement()
        self._widget.close()

    def is_click_through(self) -> bool:
        return self._mode is OverlayInteractionMode.OVERLAY

    def set_click_through(self, enabled: bool) -> None:
        self.set_interaction_mode(
            OverlayInteractionMode.OVERLAY
            if enabled
            else OverlayInteractionMode.ARRANGE
        )

    def set_arrange_mode(self, enabled: bool) -> None:
        self.set_interaction_mode(
            OverlayInteractionMode.ARRANGE
            if enabled
            else OverlayInteractionMode.OVERLAY
        )

    def set_interaction_mode(self, mode: OverlayInteractionMode | str) -> None:
        """Atomically apply the input, visual, and interaction state."""
        target = OverlayInteractionMode(mode)
        click_through = target is OverlayInteractionMode.OVERLAY
        self._set_qt_input_transparency(click_through)
        # Qt's window flag and AppKit's ignoresMouseEvents are independent on
        # macOS. Always set both, including an idempotent request, so a native
        # panel recreated by Qt cannot leave the two layers out of sync.
        set_macos_overlay_click_through(self, enabled=click_through)
        self._mode = target
        if target is OverlayInteractionMode.ARRANGE:
            self._widget.update()
        else:
            self.end_pointer_interaction()
            self._widget.update()
            self._save_placement()

    def is_arrange_mode(self) -> bool:
        return self._mode is OverlayInteractionMode.ARRANGE

    @property
    def interaction_mode(self) -> OverlayInteractionMode:
        return self._mode

    def set_panel_resize_handler(
        self, handler: Callable[[int, int], None]
    ) -> None:
        self._panel_resize_handler = handler

    def begin_pointer_interaction(
        self, global_x: int, global_y: int, local_x: int, local_y: int
    ) -> None:
        if not self.is_arrange_mode():
            return
        self._pointer_interaction_active = True
        if self._is_resize_handle(local_x, local_y):
            self._resize_origin = (global_x, global_y)
            self._resize_size_start = (self._panel_width, self._panel_height)
            self._drag_origin = None
            self._drag_window_start = None
            return
        self.begin_drag(global_x, global_y)

    def move_pointer(self, global_x: int, global_y: int) -> None:
        if self._resize_origin is not None and self._resize_size_start is not None:
            self.resize_to(global_x, global_y)
            return
        self.drag_to(global_x, global_y)

    def end_pointer_interaction(self) -> None:
        should_save = self._pointer_interaction_active and self.is_arrange_mode()
        self._pointer_interaction_active = False
        self._drag_origin = None
        self._drag_window_start = None
        self._resize_origin = None
        self._resize_size_start = None
        if should_save:
            self._save_placement()

    def reset_placement(self) -> None:
        """Restore the startup panel size and the default bottom-right location."""
        self._placement_store.clear()
        width, height = self._default_panel_size
        self._set_panel_size(width, height)
        self.move_to_corner()

    @property
    def pointer_interaction_active(self) -> bool:
        return self._pointer_interaction_active

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

    def resize_to(self, global_x: int, global_y: int) -> None:
        if self._resize_origin is None or self._resize_size_start is None:
            return
        origin_x, origin_y = self._resize_origin
        start_width, _ = self._resize_size_start
        horizontal_delta = global_x - origin_x
        vertical_delta = (global_y - origin_y) * self._ASPECT_RATIO
        # Use the dominant drag axis. The old mixed-sign formula could turn a
        # mostly horizontal growth drag into a shrink when the cursor moved a
        # few pixels upward at the same time.
        delta_width = (
            horizontal_delta
            if abs(horizontal_delta) >= abs(vertical_delta)
            else vertical_delta
        )
        width = round(
            min(
                self._maximum_panel_width(),
                max(self._MIN_PANEL_WIDTH, start_width + delta_width),
            )
        )
        height = round(width / self._ASPECT_RATIO)
        self._set_panel_size(width, height)

    def _set_panel_size(self, width: int, height: int) -> None:
        self._panel_width = width
        self._panel_height = height
        self._widget.resize(width, height)
        self._keep_panel_within_screen()
        if self._panel_resize_handler is not None:
            self._panel_resize_handler(width, height)
        if self._last_frame is not None:
            self.show_frame(self._last_frame)

    def _is_resize_handle(self, local_x: int, local_y: int) -> bool:
        return (
            local_x >= self._panel_width - self._RESIZE_HANDLE_SIZE
            and local_y >= self._panel_height - self._RESIZE_HANDLE_SIZE
        )

    def _set_qt_input_transparency(self, enabled: bool) -> None:
        set_window_flag = getattr(self._widget, "setWindowFlag", None)
        if callable(set_window_flag):
            set_window_flag(self._qt.WindowTransparentForInput, enabled)
        else:
            base = (
                self._qt.FramelessWindowHint
                | self._qt.WindowStaysOnTopHint
                | self._qt.Tool
            )
            if enabled:
                base |= self._qt.WindowTransparentForInput
            self._widget.setWindowFlags(base)
        # Qt hides a top-level widget when a window flag changes. Re-show only
        # when necessary; repeatedly calling show() caused NSPanel churn.
        if not self._widget.isVisible():
            self._widget.show()

    def _maximum_panel_width(self) -> int:
        """Return the largest 16:9 panel that fits the usable display."""
        _, _, screen_width, screen_height = (
            self._screen_geometry or self._primary_geometry()
        )
        usable_width = max(1, screen_width - self._margin * 2)
        usable_height = max(1, screen_height - self._margin * 2)
        return max(
            self._MIN_PANEL_WIDTH,
            min(usable_width, round(usable_height * self._ASPECT_RATIO)),
        )

    def _keep_panel_within_screen(self) -> None:
        """Keep a resized panel visible instead of expanding beyond screen edges."""
        screen_x, screen_y, screen_width, screen_height = (
            self._screen_geometry or self._primary_geometry()
        )
        left = screen_x + self._margin
        top = screen_y + self._margin
        right = screen_x + screen_width - self._margin
        bottom = screen_y + screen_height - self._margin
        x, y = self._position
        x = min(max(x, left), right - self._panel_width)
        y = min(max(y, top), bottom - self._panel_height)
        self._move(x, y)

    def _restore_placement(self) -> None:
        placement = self._placement_store.load()
        if placement is None:
            self.move_to_corner()
            return
        width = min(
            self._maximum_panel_width(),
            max(self._MIN_PANEL_WIDTH, placement.width),
        )
        height = round(width / self._ASPECT_RATIO)
        self._panel_width = width
        self._panel_height = height
        self._widget.resize(width, height)
        self._position = (placement.x, placement.y)
        self._keep_panel_within_screen()

    def _save_placement(self) -> None:
        # Qt can apply a native resize after the pointer callback. Persist the
        # widget's actual geometry rather than only our last requested size.
        x, y = self._position
        width, height = self._panel_width, self._panel_height
        geometry = getattr(self._widget, "geometry", None)
        if callable(geometry):
            rect = geometry()
            x, y = rect.x(), rect.y()
            width, height = rect.width(), rect.height()
        self._placement_store.save(
            OverlayPlacement(
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    def _paint_overlay(self, widget) -> None:
        """Paint pet and arrange guidance in the same live widget rect."""
        if self._qpainter is None:
            return
        painter = self._qpainter(widget)
        try:
            if self._qcolor is not None:
                source = getattr(self._qpainter, "CompositionMode_Source", None)
                source_over = getattr(
                    self._qpainter, "CompositionMode_SourceOver", None
                )
                composition_mode = getattr(self._qpainter, "CompositionMode", None)
                if source is None and composition_mode is not None:
                    source = composition_mode.CompositionMode_Source
                if source_over is None and composition_mode is not None:
                    source_over = composition_mode.CompositionMode_SourceOver
                if source is not None and source_over is not None:
                    painter.setCompositionMode(source)
                    painter.fillRect(widget.rect(), self._qcolor(0, 0, 0, 0))
                    painter.setCompositionMode(source_over)
            if self.is_arrange_mode() and self._qcolor is not None:
                painter.fillRect(widget.rect(), self._qcolor(20, 24, 28, 88))
            if self._pixmap is not None:
                # QRect is the current top-level size, so the cutout grows and
                # shrinks with the panel even before the next video frame.
                painter.drawPixmap(widget.rect(), self._pixmap)
            if self.is_arrange_mode() and self._qpen is not None and self._qcolor is not None:
                pen = self._qpen(self._qcolor(110, 235, 180, 235))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRoundedRect(widget.rect().adjusted(1, 1, -2, -2), 8, 8)
        finally:
            painter.end()

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
        return self.display_missing() or fresh

    def display_missing(self) -> ProcessedFrame | None:
        """Advance the hold/fade animation while no source frame is available."""
        if self._last_good is None:
            return None
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


class TemporalMatteRenderer:
    def __init__(
        self,
        *,
        temporal_weight: float = 0.25,
        morphology_radius: int = 1,
    ) -> None:
        self._mask_smoother = TemporalMaskSmoother(
            temporal_weight=temporal_weight,
            morphology_radius=morphology_radius,
        )

    def __call__(
        self, frame, tracking, *, feather_radius: float
    ) -> ProcessedFrame:
        return render_pet_matte(
            frame,
            tracking,
            feather_radius=feather_radius,
            mask_smoother=self._mask_smoother,
        )


class OverlayStreamStatus:
    """Thread-safe stream state shared by the decoder and Qt tray."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "Connecting"
        self._detail: str | None = None

    def set_live(self) -> None:
        self._set("Live")

    def set_reconnecting(self, detail: str | None = None) -> None:
        self._set("Reconnecting", detail)

    def label(self) -> str:
        with self._lock:
            suffix = f" ({self._detail})" if self._detail else ""
            return f"{self._state}{suffix}"

    def _set(self, state: str, detail: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._detail = detail


class _OverlayTray:
    def __init__(
        self,
        icon,
        menu,
        status_action,
        quality_action,
        arrange_action,
        reset_placement_action,
        quit_action,
        window,
        on_quit,
        stream_status,
    ):
        self._icon = icon
        self._menu = menu
        self._status_action = status_action
        self._quality_action = quality_action
        self._arrange_action = arrange_action
        self._reset_placement_action = reset_placement_action
        self._quit_action = quit_action
        self._window = window
        self._on_quit = on_quit
        self._stream_status = stream_status
        self._quitting = False

    @property
    def icon(self):
        return self._icon

    def toggle_arrange_mode(self) -> None:
        self.set_arrange_mode(not self._window_is_arranging())

    def set_arrange_mode(self, enabled: bool) -> None:
        arrange_on = bool(enabled)
        # The window owns this state. Update the menu only after the native
        # input policy has been applied successfully, so the label cannot lie.
        set_arrange_mode = getattr(self._window, "set_arrange_mode", None)
        if callable(set_arrange_mode):
            set_arrange_mode(arrange_on)
        else:
            self._window.set_click_through(not arrange_on)
        self._arrange_action.setText(
            "Return to Overlay Mode" if arrange_on else "Arrange Overlay…"
        )

    def set_menu_visibility_handlers(self, *, on_open, on_close) -> None:
        """Pause frame work while macOS is tracking the status-menu click."""
        about_to_show = getattr(self._menu, "aboutToShow", None)
        about_to_hide = getattr(self._menu, "aboutToHide", None)
        if about_to_show is not None:
            about_to_show.connect(on_open)
        if about_to_hide is not None:
            about_to_hide.connect(on_close)

    def refresh_status(self) -> None:
        label = self._stream_status.label()
        self._status_action.setText(f"Stream: {label}")
        self._icon.setToolTip(f"WithNuri — {label}")

    def reset_placement(self) -> None:
        reset = getattr(self._window, "reset_placement", None)
        if callable(reset):
            reset()

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._on_quit()

    def _window_is_arranging(self) -> bool:
        is_arrange_mode = getattr(self._window, "is_arrange_mode", None)
        if callable(is_arrange_mode):
            return bool(is_arrange_mode())
        return not self._window.is_click_through()


def _make_tray_icon(modules):
    # A null QIcon renders as an invisible/blank slot in the menu bar, so draw a
    # small solid swatch when the drawing classes are available. Injected fake
    # qt_modules (tests) omit QPixmap/QColor and fall back to the empty icon.
    if "QPixmap" in modules and "QColor" in modules:
        pixmap = modules["QPixmap"](22, 22)
        pixmap.fill(modules["QColor"](0, 200, 120))
        return modules["QIcon"](pixmap)
    return modules["QIcon"]()


def build_overlay_tray(
    window,
    *,
    on_quit,
    quality_profile: str = "balanced",
    stream_status: OverlayStreamStatus | None = None,
    qt_modules: dict | None = None,
):
    modules = qt_modules or _load_tray_qt_modules()
    icon = modules["QSystemTrayIcon"](_make_tray_icon(modules))
    menu = modules["QMenu"]()
    stream_status = stream_status or OverlayStreamStatus()

    status_action = modules["QAction"]("Stream: Connecting")
    status_action.setEnabled(False)
    quality_action = modules["QAction"](
        f"Quality: {_quality_profile_label(quality_profile)} (restart to change)"
    )
    quality_action.setEnabled(False)
    arrange_action = modules["QAction"]("Arrange Overlay…")
    reset_placement_action = modules["QAction"]("Reset Overlay Placement")
    quit_action = modules["QAction"]("Quit")

    menu.addAction(status_action)
    menu.addAction(quality_action)
    menu.addSeparator()
    menu.addAction(arrange_action)
    menu.addAction(reset_placement_action)
    menu.addSeparator()
    menu.addAction(quit_action)
    icon.setContextMenu(menu)
    icon.show()

    tray = _OverlayTray(
        icon,
        menu,
        status_action,
        quality_action,
        arrange_action,
        reset_placement_action,
        quit_action,
        window,
        on_quit,
        stream_status,
    )
    tray.refresh_status()
    arrange_action.triggered.connect(lambda *_: tray.toggle_arrange_mode())
    reset_placement_action.triggered.connect(lambda *_: tray.reset_placement())
    quit_action.triggered.connect(lambda *_: tray.quit())
    return tray


def _quality_profile_label(profile: str) -> str:
    return profile.replace("-", " ").title()


def apply_macos_overlay_chrome(window, *, appkit="__auto__") -> bool:
    if appkit == "__auto__":
        appkit = _load_appkit()
    if appkit is None:
        return False
    view = appkit.view_for(int(window.widget.winId()))
    ns_window = view.window()
    # Floating stays above ordinary application windows but below status-item
    # menus. NSStatusWindowLevel can visually/input-wise compete with the menu
    # bar popover on macOS.
    level = getattr(appkit, "NSFloatingWindowLevel", appkit.NSStatusWindowLevel)
    ns_window.setLevel_(level)
    ns_window.setCollectionBehavior_((1 << 0) | (1 << 8) | (1 << 4))
    ns_window.setHasShadow_(False)
    return set_macos_overlay_click_through(
        window, enabled=window.is_click_through(), appkit=appkit
    )


def set_macos_overlay_click_through(
    window, *, enabled: bool, appkit="__auto__"
) -> bool:
    """Keep AppKit mouse handling in sync with Qt's click-through flag."""
    if appkit == "__auto__":
        appkit = _load_appkit()
    if appkit is None:
        return False
    win_id = getattr(window.widget, "winId", None)
    if not callable(win_id):
        return False
    view = appkit.view_for(int(win_id()))
    view.window().setIgnoresMouseEvents_(enabled)
    return True


def _load_appkit():
    import sys

    if sys.platform != "darwin":
        return None
    try:
        import objc
        from AppKit import NSFloatingWindowLevel, NSStatusWindowLevel
    except ImportError:
        return None

    # A class body cannot see the enclosing function's locals, so writing
    # `NSStatusWindowLevel = NSStatusWindowLevel` in the class body raises
    # NameError. Set it as an instance attribute from function scope instead.
    # (view_for is a method, which *does* close over `objc`, so it resolves.)
    class _AppKitAdapter:
        @staticmethod
        def view_for(win_id):
            return objc.objc_object(c_void_p=win_id)

    adapter = _AppKitAdapter()
    adapter.NSStatusWindowLevel = NSStatusWindowLevel
    adapter.NSFloatingWindowLevel = NSFloatingWindowLevel
    return adapter


def run_pet_overlay_qt(
    decoder,
    *,
    tracker,
    window,
    fade=None,
    quality_profile: str = "balanced",
    **kwargs,
):
    from PySide6.QtCore import QTimer

    app = window._app

    try:
        apply_macos_overlay_chrome(window)
    except Exception:
        pass

    # Build the system tray (Quit + arrange-mode toggle) and tie its lifetime
    # to the window so Qt does not garbage-collect the QSystemTrayIcon, which
    # would silently drop the menu. on_quit stops the event loop.
    stream_status = OverlayStreamStatus()
    window._tray = build_overlay_tray(
        window,
        on_quit=app.quit,
        quality_profile=quality_profile,
        stream_status=stream_status,
    )

    def app_runner(tick):
        timer = QTimer()

        def on_timeout():
            keep_running = tick()
            window._tray.refresh_status()
            if not keep_running:
                timer.stop()
                app.quit()

        timer.timeout.connect(on_timeout)

        def pause_for_menu() -> None:
            if timer.isActive():
                timer.stop()

        def resume_after_menu() -> None:
            if not window.should_close():
                timer.start(16)

        # Inference runs on this Qt thread today. Pausing the frame pump while
        # the system tray menu is open makes its actions deterministic instead
        # of competing with an immediately recurring 0ms timer.
        window._tray.set_menu_visibility_handlers(
            on_open=pause_for_menu, on_close=resume_after_menu
        )
        timer.start(16)
        app.exec()

    if fade is None:
        fade = FadeController()
    result, interrupted = _run_with_sigint_handler(
        lambda: run_pet_overlay(
            decoder,
            tracker=tracker,
            window=window,
            app_runner=app_runner,
            fade=fade,
            stream_status=stream_status,
            **kwargs,
        ),
        quit_app=app.quit,
    )
    return replace(result, interrupted=True) if interrupted else result


def _run_with_sigint_handler(run, *, quit_app, signal_module=signal):
    """Make Ctrl-C stop the Qt loop instead of leaving its timers running."""
    interrupted = threading.Event()

    def handle_sigint(_signum, _frame) -> None:
        interrupted.set()
        quit_app()

    previous_handler = signal_module.getsignal(signal_module.SIGINT)
    signal_module.signal(signal_module.SIGINT, handle_sigint)
    try:
        result = run()
    finally:
        signal_module.signal(signal_module.SIGINT, previous_handler)
    return result, interrupted.is_set()


def run_pet_overlay(
    decoder,
    *,
    tracker,
    window,
    app_runner,
    renderer=None,
    fade=None,
    feather_radius: float = 1.5,
    max_frames: int | None = None,
    sleep=time.sleep,
    idle_seconds: float = 0.005,
    collect_diagnostics: bool = False,
    clock=time.monotonic,
    stabilizer: PetDisplayStabilizer | None = None,
    crop_controller: PetCropController | None = None,
    inference_gate: IdleInferenceGate | None = None,
    stream_status: OverlayStreamStatus | None = None,
    reconnect_delay_seconds: float | None = None,
    on_stream_status: Callable[[str, str | None], None] | None = None,
) -> PreviewResult:
    if reconnect_delay_seconds is not None and reconnect_delay_seconds < 0:
        raise ValueError("reconnect_delay_seconds must not be negative")
    queue: LatestFrameQueue = LatestFrameQueue()
    stop = threading.Event()
    reconnecting = threading.Event()
    stream_reset_pending = threading.Event()
    producer_error: list[BaseException] = []
    diagnostics = (
        PipelineDiagnosticsCollector(clock=clock) if collect_diagnostics else None
    )
    stabilizer = stabilizer or PetDisplayStabilizer()
    renderer = renderer or TemporalMatteRenderer()

    def reset_for_reconnect() -> None:
        for component in (tracker, stabilizer, renderer, crop_controller, inference_gate):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def report_stream_status(status: str, detail: str | None = None) -> None:
        if on_stream_status is not None:
            on_stream_status(status, detail)

    def produce() -> None:
        try:
            was_reconnecting = False
            has_received_frame = False
            while not stop.is_set():
                frame_iter = None
                stream_detail = "stream ended"
                try:
                    frame_iter = decoder.iter_frames()
                    for raw_frame in frame_iter:
                        if stop.is_set():
                            break
                        if not has_received_frame:
                            if stream_status is not None:
                                stream_status.set_live()
                            has_received_frame = True
                        if was_reconnecting:
                            stream_reset_pending.set()
                            reconnecting.clear()
                            was_reconnecting = False
                            if stream_status is not None:
                                stream_status.set_live()
                            report_stream_status("live")
                        if diagnostics is not None:
                            diagnostics.record_decoded_frame()
                        queue.put(raw_frame)
                except DecodeToolMissing:
                    raise
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    stream_detail = str(exc) or type(exc).__name__
                    if reconnect_delay_seconds is None:
                        raise
                finally:
                    if frame_iter is not None:
                        frame_iter.close()

                if stop.is_set():
                    break
                if reconnect_delay_seconds is None:
                    break
                reconnecting.set()
                was_reconnecting = True
                if stream_status is not None:
                    stream_status.set_reconnecting(stream_detail)
                report_stream_status("reconnecting", stream_detail)
                stop.wait(reconnect_delay_seconds)
        except BaseException as exc:  # surfaced to the consumer after join
            producer_error.append(exc)
        finally:
            stop.set()

    thread = threading.Thread(target=produce, name="withnuri-overlay-decode", daemon=True)

    state = {"frames_rendered": 0, "visible": 0, "last_visible": False}
    last_reconnect_present_at: float | None = None

    def tick() -> bool:
        nonlocal last_reconnect_present_at
        if window.should_close():
            stop.set()
            return False
        raw_frame = queue.latest()
        if raw_frame is None:
            now = clock()
            if (
                reconnecting.is_set()
                and fade is not None
                and (
                    last_reconnect_present_at is None
                    or now - last_reconnect_present_at >= 1 / 30
                )
            ):
                frame_to_show = fade.display_missing()
                if frame_to_show is not None:
                    if crop_controller is not None:
                        frame_to_show = crop_controller.crop(frame_to_show, [])
                    window.show_frame(frame_to_show)
                last_reconnect_present_at = now
            if stop.is_set():
                return False
            sleep(idle_seconds)
            return True
        if inference_gate is not None and not inference_gate.should_infer(
            raw_frame.timestamp_seconds
        ):
            if diagnostics is not None:
                diagnostics.record_skipped_inference_frame()
            return not stop.is_set()
        if stream_reset_pending.is_set():
            # The tracker and render state are consumed on the Qt/main thread.
            # Resetting them here avoids racing a still-processing frame from
            # the previous RTSP connection in the decode producer thread.
            reset_for_reconnect()
            stream_reset_pending.clear()
        processing_started_at = clock()
        raw_tracking = tracker.track(raw_frame)
        inference_seconds = clock() - processing_started_at
        tracking = stabilizer.stabilize(raw_tracking)
        if inference_gate is not None:
            inference_gate.record_inference(
                timestamp_seconds=raw_frame.timestamp_seconds,
                pet_is_active=any(track.detected for track in raw_tracking.tracks)
                or bool(tracking.tracks),
            )
        fresh = renderer(raw_frame, tracking, feather_radius=feather_radius)
        # Cached tracks deliberately retain their identity but do not represent
        # a current segmentation. Start the grace-period/fade immediately when
        # no fresh pet mask was detected, rather than drawing stale masks over
        # the latest camera frame.
        has_pet = any(track.detected for track in tracking.tracks)
        frame_to_show = fade.display_frame(fresh, has_pet) if fade else fresh
        if crop_controller is not None:
            frame_to_show = crop_controller.crop(frame_to_show, tracking.tracks)
        render_seconds = clock() - processing_started_at - inference_seconds
        present_started_at = clock()
        window.show_frame(frame_to_show)
        present_seconds = clock() - present_started_at
        state["frames_rendered"] += 1
        state["last_visible"] = has_pet
        if has_pet:
            state["visible"] += 1
        if diagnostics is not None:
            diagnostics.record_processed_frame(
                queue_delay_seconds=processing_started_at - raw_frame.timestamp_seconds,
                inference_seconds=inference_seconds,
                render_seconds=render_seconds,
                present_seconds=present_seconds,
                detected=any(track.detected for track in raw_tracking.tracks),
                track_ids={
                    track.track_id
                    for track in raw_tracking.tracks
                    if track.detected and track.track_id is not None
                },
            )
        if window.should_close():
            stop.set()
            return False
        if max_frames is not None and state["frames_rendered"] >= max_frames:
            stop.set()
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
        diagnostics=(
            diagnostics.snapshot(
                dropped_frame_count=queue.dropped_frame_count,
                display_metrics=stabilizer.metrics(),
            )
            if diagnostics is not None
            else None
        ),
    )


def _load_tray_qt_modules() -> dict:
    try:
        from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
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
        "QPixmap": QPixmap,
        "QColor": QColor,
    }


def _load_overlay_qt_modules() -> dict:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
        from PySide6.QtWidgets import QApplication, QWidget
    except ImportError as exc:
        raise OverlayWindowUnavailable(
            "PySide6 is required for the transparent overlay"
        ) from exc
    return {
        "QApplication": QApplication,
        "QWidget": QWidget,
        "QImage": QImage,
        "QPixmap": QPixmap,
        "QPainter": QPainter,
        "QPen": QPen,
        "QColor": QColor,
        "Qt": Qt,
    }
