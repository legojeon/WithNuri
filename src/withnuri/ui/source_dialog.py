"""Small launch-time source picker used by the demo app entry point."""

from __future__ import annotations

from withnuri.streaming.sources import (
    DEFAULT_DEMO_SOURCE_URL,
    StreamSourceStore,
    normalize_rtsp_url,
)


def choose_overlay_source(
    initial_url: str | None = None,
    *,
    store: StreamSourceStore | None = None,
) -> str | None:
    """Return a selected RTSP URL, or ``None`` when the user cancels.

    Keeping this dialog at launch deliberately avoids replacing a live decoder,
    tracker, and overlay window from a tray-menu callback. The existing Qt app
    instance is reused by the overlay created immediately afterwards.
    """
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QVBoxLayout,
        )
    except ImportError as exc:
        raise RuntimeError("PySide6 is required to choose a stream source") from exc

    store = store or StreamSourceStore()
    _app = QApplication.instance() or QApplication([])
    # The source dialog is temporarily the only normal top-level window. Keep
    # Qt from treating its close as an application shutdown before the
    # transparent overlay has been created.
    set_quit_on_last_window_closed = getattr(_app, "setQuitOnLastWindowClosed", None)
    if callable(set_quit_on_last_window_closed):
        set_quit_on_last_window_closed(False)
    dialog = QDialog()
    dialog.setWindowTitle("WithNuri — Choose Stream")
    dialog.setModal(True)
    dialog.setMinimumWidth(460)

    layout = QVBoxLayout(dialog)
    intro = QLabel("Choose a saved camera stream or enter an RTSP address.")
    intro.setWordWrap(True)
    layout.addWidget(intro)

    form = QFormLayout()
    presets: list[tuple[str, str]] = [
        ("Local demo relay", DEFAULT_DEMO_SOURCE_URL),
    ]
    for url in store.load_recent():
        if url != DEFAULT_DEMO_SOURCE_URL:
            presets.append((f"Recent — {url}", url))
    if initial_url:
        try:
            normalized_initial = normalize_rtsp_url(initial_url)
        except ValueError:
            normalized_initial = initial_url
        if all(url != normalized_initial for _, url in presets):
            presets.append((f"Requested — {normalized_initial}", normalized_initial))

    source_box = QComboBox()
    for label, url in presets:
        source_box.addItem(label, url)
    url_field = QLineEdit(initial_url or presets[0][1])
    url_field.setPlaceholderText("rtsp://host:8554/stream-name")
    form.addRow("Saved source", source_box)
    form.addRow("RTSP address", url_field)
    layout.addLayout(form)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Open
    )
    layout.addWidget(buttons)

    def select_preset(index: int) -> None:
        url = source_box.itemData(index)
        if isinstance(url, str):
            url_field.setText(url)

    def accept() -> None:
        try:
            selected = normalize_rtsp_url(url_field.text())
        except ValueError as exc:
            QMessageBox.warning(dialog, "Invalid stream address", str(exc))
            return
        url_field.setText(selected)
        dialog.accept()

    source_box.currentIndexChanged.connect(select_preset)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(accept)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return store.remember(url_field.text())
