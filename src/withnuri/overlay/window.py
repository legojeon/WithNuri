from withnuri.pipeline.frames import ProcessedFrame
from withnuri.ui.windowing import PreviewWindowUnavailable, resize_processed_frame


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
