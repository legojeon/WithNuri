import threading
import time
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
import signal
import sys

from withnuri.pipeline.frames import ProcessedFrame
from withnuri.pipeline.inference_gate import IdleInferenceGate
from withnuri.pipeline.diagnostics import PipelineDiagnosticsCollector
from withnuri.pipeline.display_stabilizer import PetDisplayStabilizer
from withnuri.pipeline.debug_overlay import render_tracking_debug_overlay
from withnuri.pipeline.matte import render_pet_matte, scale_processed_alpha
from withnuri.pipeline.mask_smoothing import TemporalMaskSmoother
from withnuri.pipeline.pet_crop import PetCropController
from withnuri.pipeline.queue import LatestFrameQueue
from withnuri.runtime import resource_path
from withnuri.streaming.decoder import DecodeToolMissing
from withnuri.overlay.placement import OverlayPlacement, OverlayPlacementStore
from withnuri.ui.windowing import (
    PreviewResult,
    PreviewWindowUnavailable,
    resize_processed_frame,
)
from withnuri.ui.debug_window import QtDebugWindow


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


class OverlayDebugPreview:
    """Switches between the transparent pet output and a live debug window."""

    def __init__(self, overlay_window, *, debug_window_factory=QtDebugWindow) -> None:
        self._overlay_window = overlay_window
        self._debug_window_factory = debug_window_factory
        self._debug_window = None

    @property
    def is_enabled(self) -> bool:
        if self._debug_window is not None and self._debug_window.should_close():
            self.disable()
        return self._debug_window is not None

    def toggle(self) -> None:
        if self.is_enabled:
            self.disable()
        else:
            window_kwargs = {
                "title": "WithNuri Debug Preview",
                "always_on_top": True,
            }
            output_geometry = getattr(self._overlay_window, "output_geometry", None)
            if isinstance(output_geometry, tuple) and len(output_geometry) == 4:
                x, y, width, height = output_geometry
                window_kwargs.update(
                    display_x=x,
                    display_y=y,
                    display_width=width,
                    display_height=height,
                )
            self._debug_window = self._debug_window_factory(**window_kwargs)
            self._overlay_window.set_output_visible(False)

    def present(self, raw_frame, tracking) -> None:
        if not self.is_enabled:
            return
        self._debug_window.show_frame(render_tracking_debug_overlay(raw_frame, tracking))

    def disable(self) -> None:
        if self._debug_window is not None:
            self._debug_window.close()
            self._debug_window = None
        self._overlay_window.set_output_visible(True)


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
        self._output_visible = True
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
        window_type = getattr(self._qt, "Window", 0)
        self._widget.setWindowFlags(
            window_type
            | self._qt.FramelessWindowHint
            | self._qt.WindowStaysOnTopHint
            | self._qt.WindowTransparentForInput
        )
        self._widget.setAttribute(self._qt.WA_TranslucentBackground, True)
        self._widget.setAttribute(self._qt.WA_NoSystemBackground, True)

        self._widget.resize(panel_width, panel_height)
        self._screen_geometry = screen_geometry
        self._position = (0, 0)
        self._restore_placement()
        self._widget.show()

    @property
    def widget(self):
        return self._widget

    @property
    def output_size(self) -> tuple[int, int]:
        """Actual visible panel dimensions for an equivalent debug preview."""
        size = getattr(self._widget, "size", None)
        if callable(size):
            current = size()
            return (current.width(), current.height())
        return (self._panel_width, self._panel_height)

    @property
    def output_geometry(self) -> tuple[int, int, int, int]:
        geometry = getattr(self._widget, "geometry", None)
        if callable(geometry):
            rect = geometry()
            return (rect.x(), rect.y(), rect.width(), rect.height())
        width, height = self.output_size
        return (self._position[0], self._position[1], width, height)

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
        if not self._output_visible:
            return False
        return not self._widget.isVisible()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._save_placement()
        self._widget.close()

    def set_output_visible(self, visible: bool) -> None:
        if visible == self._output_visible:
            return
        self._output_visible = visible
        if visible:
            self._widget.show()
            set_macos_overlay_click_through(self, enabled=self.is_click_through())
        else:
            self._widget.hide()

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
            # The Arrange guide uses translucent pixels. On a macOS
            # translucent top-level surface, a deferred update can leave that
            # backing-store content visible until a later video frame arrives.
            # Paint the fully-cleared Overlay surface immediately instead.
            repaint = getattr(self._widget, "repaint", None)
            if callable(repaint):
                repaint()
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
                getattr(self._qt, "Window", 0)
                | self._qt.FramelessWindowHint
                | self._qt.WindowStaysOnTopHint
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
                clear = self._painter_composition_mode("CompositionMode_Clear")
                source = self._painter_composition_mode("CompositionMode_Source")
                source_over = self._painter_composition_mode(
                    "CompositionMode_SourceOver"
                )
                if clear is not None:
                    # Clear ignores the brush alpha and therefore reliably
                    # erases stale Arrange-mode pixels on a transparent
                    # backing store. Source + transparent is only a fallback
                    # for minimal/fake QPainter implementations.
                    painter.setCompositionMode(clear)
                    painter.fillRect(widget.rect(), self._qcolor(0, 0, 0, 0))
                elif source is not None:
                    painter.setCompositionMode(source)
                    painter.fillRect(widget.rect(), self._qcolor(0, 0, 0, 0))
                if source_over is not None:
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

    def _painter_composition_mode(self, name: str):
        mode = getattr(self._qpainter, name, None)
        if mode is not None:
            return mode
        enum = getattr(self._qpainter, "CompositionMode", None)
        return getattr(enum, name, None) if enum is not None else None

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

    def reset(self) -> None:
        self._last_good = None
        self._lost_at = None

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

    def is_live(self) -> bool:
        with self._lock:
            return self._state == "Live"

    def hint(self) -> str:
        with self._lock:
            if self._state == "Connecting":
                return "Waiting for the camera stream…"
            if self._state == "Reconnecting":
                return "Check MediaMTX and the camera publisher."
            return ""

    def _set(self, state: str, detail: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._detail = detail


class OverlayPetStatus:
    """Thread-safe display state that distinguishes stream and pet health."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label = "Waiting for stream"

    def set_searching(self) -> None:
        self._set("Searching for pet…")

    def set_detected(self, detected: bool) -> None:
        self._set("Pet detected" if detected else "No pet detected")

    def set_waiting_for_stream(self) -> None:
        self._set("Waiting for stream")

    def label(self) -> str:
        with self._lock:
            return self._label

    def _set(self, label: str) -> None:
        with self._lock:
            self._label = label


class _OverlayTray:
    def __init__(
        self,
        icon,
        menu,
        status_action,
        source_action,
        pet_action,
        hint_action,
        quality_action,
        debug_action,
        arrange_action,
        reset_placement_action,
        quit_action,
        window,
        on_quit,
        stream_status,
        pet_status,
        source_label,
        debug_preview,
    ):
        self._icon = icon
        self._menu = menu
        self._status_action = status_action
        self._source_action = source_action
        self._pet_action = pet_action
        self._hint_action = hint_action
        self._quality_action = quality_action
        self._debug_action = debug_action
        self._arrange_action = arrange_action
        self._reset_placement_action = reset_placement_action
        self._quit_action = quit_action
        self._window = window
        self._on_quit = on_quit
        self._stream_status = stream_status
        self._pet_status = pet_status
        self._source_label = source_label
        self._debug_preview = debug_preview
        self._quitting = False

    @property
    def icon(self):
        return self._icon

    def show(self) -> None:
        """Register the status item after Qt owns the running event loop."""
        self._icon.show()
        self.refresh_status()

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
        self._source_action.setText(f"Source: {self._source_label}")
        pet_label = (
            self._pet_status.label()
            if self._stream_status.is_live()
            else "Waiting for stream"
        )
        self._pet_action.setText(f"Pet: {pet_label}")
        hint = self._stream_status.hint()
        self._hint_action.setText(hint)
        self._hint_action.setEnabled(bool(hint))
        set_visible = getattr(self._hint_action, "setVisible", None)
        if callable(set_visible):
            set_visible(bool(hint))
        self._icon.setToolTip(f"WithNuri — {label}; {pet_label}")
        self._debug_action.setText(
            "Hide Debug Preview"
            if self._debug_preview.is_enabled
            else "Show Debug Preview"
        )

    def toggle_debug_preview(self) -> None:
        self._debug_preview.toggle()
        self.refresh_status()

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
    icon = modules["QIcon"](str(resource_path("assets/withnuri-tray.svg")))
    is_null = getattr(icon, "isNull", None)
    if not callable(is_null) or not is_null():
        return icon

    # Keep a visible fallback for incomplete Qt installations. Injected fake
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
    source_label: str = "Current source",
    stream_status: OverlayStreamStatus | None = None,
    pet_status: OverlayPetStatus | None = None,
    debug_preview: OverlayDebugPreview | None = None,
    qt_modules: dict | None = None,
):
    modules = qt_modules or _load_tray_qt_modules()
    icon = modules["QSystemTrayIcon"](_make_tray_icon(modules))
    menu = modules["QMenu"]()
    stream_status = stream_status or OverlayStreamStatus()
    pet_status = pet_status or OverlayPetStatus()
    debug_preview = debug_preview or OverlayDebugPreview(window)

    status_action = modules["QAction"]("Stream: Connecting")
    status_action.setEnabled(False)
    source_action = modules["QAction"](f"Source: {source_label}")
    source_action.setEnabled(False)
    pet_action = modules["QAction"]("Pet: Waiting for stream")
    pet_action.setEnabled(False)
    hint_action = modules["QAction"]("")
    hint_action.setEnabled(False)
    quality_action = modules["QAction"](
        f"Quality: {_quality_profile_label(quality_profile)}"
    )
    quality_action.setEnabled(False)
    debug_action = modules["QAction"]("Show Debug Preview")
    arrange_action = modules["QAction"]("Arrange Overlay…")
    reset_placement_action = modules["QAction"]("Reset Overlay Placement")
    quit_action = modules["QAction"]("Quit")

    menu.addAction(status_action)
    menu.addAction(source_action)
    menu.addAction(pet_action)
    menu.addAction(hint_action)
    menu.addAction(quality_action)
    menu.addSeparator()
    menu.addAction(debug_action)
    menu.addSeparator()
    menu.addAction(arrange_action)
    menu.addAction(reset_placement_action)
    menu.addSeparator()
    menu.addAction(quit_action)
    icon.setContextMenu(menu)

    tray = _OverlayTray(
        icon,
        menu,
        status_action,
        source_action,
        pet_action,
        hint_action,
        quality_action,
        debug_action,
        arrange_action,
        reset_placement_action,
        quit_action,
        window,
        on_quit,
        stream_status,
        pet_status,
        source_label,
        debug_preview,
    )
    tray.show()
    debug_action.triggered.connect(lambda *_: tray.toggle_debug_preview())
    arrange_action.triggered.connect(lambda *_: tray.toggle_arrange_mode())
    reset_placement_action.triggered.connect(lambda *_: tray.reset_placement())
    quit_action.triggered.connect(lambda *_: tray.quit())
    return tray


def _quality_profile_label(profile: str) -> str:
    return profile.replace("-", " ").title()


def apply_macos_overlay_chrome(window, *, appkit="__auto__") -> bool:
    """Apply persistent native properties without changing app activation."""
    native = _macos_native_window(window, appkit=appkit)
    if native is None:
        return False
    ns_window, appkit = native
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


def bring_macos_overlay_to_front(window, *, appkit="__auto__") -> bool:
    """Order a visible overlay after Qt has entered its event loop.

    This runs from a queued callback after the source dialog has closed and
    Qt has finished showing the normal top-level transparent window.
    """
    widget = window.widget
    show = getattr(widget, "show", None)
    if callable(show):
        show()
    raise_window = getattr(widget, "raise_", None)
    if callable(raise_window):
        raise_window()

    native = _macos_native_window(window, appkit=appkit)
    if native is None:
        return False
    ns_window, appkit = native
    apply_macos_overlay_chrome(window, appkit=appkit)
    order_front_regardless = getattr(ns_window, "orderFrontRegardless", None)
    if callable(order_front_regardless):
        order_front_regardless()
    return True


def _macos_native_window(window, *, appkit="__auto__"):
    if appkit == "__auto__":
        appkit = _load_appkit()
    if appkit is None:
        return None
    view = appkit.view_for(int(window.widget.winId()))
    ns_window = view.window()
    if ns_window is None:
        raise RuntimeError("WithNuri overlay has no native macOS window")
    return ns_window, appkit


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
    source_label: str = "Current source",
    **kwargs,
):
    from PySide6.QtCore import QTimer

    app = window._app
    # Debug Preview temporarily hides the transparent output. Closing that
    # preview must return to the overlay, not let Qt quit because it believes
    # the last visible top-level window was closed. Explicit Quit remains the
    # tray action's responsibility.
    _keep_qt_running_without_visible_windows(app)

    _apply_overlay_chrome_safely(window)

    # Build the system tray (Quit + arrange-mode toggle) and tie its lifetime
    # to the window so Qt does not garbage-collect the QSystemTrayIcon, which
    # would silently drop the menu. on_quit stops the event loop.
    stream_status = OverlayStreamStatus()
    pet_status = OverlayPetStatus()
    debug_preview = OverlayDebugPreview(window)
    window._tray = build_overlay_tray(
        window,
        on_quit=app.quit,
        quality_profile=quality_profile,
        source_label=source_label,
        stream_status=stream_status,
        pet_status=pet_status,
        debug_preview=debug_preview,
    )

    def app_runner(tick):
        timer = QTimer()

        # The source dialog closes before this event loop begins. Reassert the
        # normal-window level after Qt completes that modal transition.
        QTimer.singleShot(0, window._tray.show)
        QTimer.singleShot(0, lambda: _bring_overlay_to_front_safely(window))
        QTimer.singleShot(150, lambda: _bring_overlay_to_front_safely(window))

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
            pet_status=pet_status,
            debug_preview=debug_preview,
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


def _keep_qt_running_without_visible_windows(app) -> None:
    set_quit_on_last_window_closed = getattr(app, "setQuitOnLastWindowClosed", None)
    if callable(set_quit_on_last_window_closed):
        set_quit_on_last_window_closed(False)


def _apply_overlay_chrome_safely(window) -> bool:
    try:
        return apply_macos_overlay_chrome(window)
    except Exception as exc:
        print(f"Overlay native chrome setup failed: {exc}", file=sys.stderr)
        return False


def _bring_overlay_to_front_safely(window) -> bool:
    try:
        return bring_macos_overlay_to_front(window)
    except Exception as exc:
        print(f"Overlay native front-order failed: {exc}", file=sys.stderr)
        return False


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
    pet_status: OverlayPetStatus | None = None,
    debug_preview: OverlayDebugPreview | None = None,
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
                            if pet_status is not None:
                                pet_status.set_searching()
                            has_received_frame = True
                        if was_reconnecting:
                            stream_reset_pending.set()
                            reconnecting.clear()
                            was_reconnecting = False
                            if stream_status is not None:
                                stream_status.set_live()
                            if pet_status is not None:
                                pet_status.set_searching()
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
                if pet_status is not None:
                    pet_status.set_waiting_for_stream()
                report_stream_status("reconnecting", stream_detail)
                stop.wait(reconnect_delay_seconds)
        except BaseException as exc:  # surfaced to the consumer after join
            producer_error.append(exc)
        finally:
            stop.set()

    thread = threading.Thread(target=produce, name="withnuri-overlay-decode", daemon=True)

    state = {"frames_rendered": 0, "visible": 0, "last_visible": False}
    last_reconnect_present_at: float | None = None
    debug_mode_active = False

    def reset_for_output_mode_switch() -> None:
        for component in (renderer, crop_controller, fade):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def tick() -> bool:
        nonlocal debug_mode_active, last_reconnect_present_at
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
                    if debug_preview is None or not debug_preview.is_enabled:
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
        debug_enabled = debug_preview is not None and debug_preview.is_enabled
        if debug_enabled != debug_mode_active:
            reset_for_output_mode_switch()
            debug_mode_active = debug_enabled
        if debug_enabled:
            debug_preview.present(raw_frame, tracking)
        if inference_gate is not None:
            inference_gate.record_inference(
                timestamp_seconds=raw_frame.timestamp_seconds,
                pet_is_active=any(track.detected for track in raw_tracking.tracks)
                or bool(tracking.tracks),
            )
        has_pet = any(track.detected for track in tracking.tracks)
        if pet_status is not None:
            pet_status.set_detected(has_pet)
        present_started_at = clock()
        if not debug_enabled:
            fresh = renderer(raw_frame, tracking, feather_radius=feather_radius)
            # Cached tracks deliberately retain their identity but do not represent
            # a current segmentation. Start the grace-period/fade immediately when
            # no fresh pet mask was detected, rather than drawing stale masks over
            # the latest camera frame.
            frame_to_show = fade.display_frame(fresh, has_pet) if fade else fresh
            if crop_controller is not None:
                frame_to_show = crop_controller.crop(frame_to_show, tracking.tracks)
            window.show_frame(frame_to_show)
        render_seconds = clock() - processing_started_at - inference_seconds
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
