"""Small launch-time source picker used by the demo app entry point."""

from __future__ import annotations

from dataclasses import dataclass

from withnuri.streaming.sources import (
    DEFAULT_DEMO_SOURCE_URL,
    QUALITY_PROFILES,
    StreamSourceStore,
    is_reachable_lan_ipv4,
    local_lan_ipv4,
    normalize_rtsp_url,
)


@dataclass(frozen=True)
class OverlayLaunchSelection:
    url: str
    quality: str
    start_local_demo: bool = False
    start_local_relay: bool = False


def choose_overlay_source(
    initial_url: str | None = None,
    initial_quality: str | None = None,
    *,
    store: StreamSourceStore | None = None,
) -> OverlayLaunchSelection | None:
    """Return launch settings, or ``None`` when the user cancels.

    Keeping this dialog at launch deliberately avoids replacing a live decoder,
    tracker, and overlay window from a tray-menu callback. The existing Qt app
    instance is reused by the overlay created immediately afterwards.
    """
    try:
        from PySide6.QtCore import Qt
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
    # This is a compact launch form, not a workspace. Keeping its dimensions
    # fixed prevents a maximized dialog from leaving the controls stranded in
    # one corner while still giving a full RTSP URL comfortable room.
    dialog.setFixedSize(640, 390)
    dialog.setSizeGripEnabled(False)

    layout = QVBoxLayout(dialog)
    intro = QLabel("Choose the included demo, receive Moblin, or connect an RTSP camera stream.")
    intro.setWordWrap(True)
    intro.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(intro)

    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    presets: list[tuple[str, str]] = []
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

    if not presets:
        presets.append(("No saved streams — enter an address below", ""))

    mode_box = QComboBox()
    mode_box.setFixedWidth(460)
    mode_box.addItem("Play local demo — included dog video", "local-demo")
    mode_box.addItem("Receive Moblin — broadcast from your phone", "local-relay")
    mode_box.addItem("Connect existing RTSP stream", "rtsp")

    source_box = QComboBox()
    source_box.setFixedWidth(460)
    for label, url in presets:
        source_box.addItem(label, url)
    default_url = initial_url or DEFAULT_DEMO_SOURCE_URL
    url_field = QLineEdit(default_url)
    url_field.setFixedWidth(460)
    url_field.setPlaceholderText("rtsp://host:8554/stream-name")
    publish_host_field = QLineEdit(local_lan_ipv4() or "")
    publish_host_field.setFixedWidth(460)
    publish_host_field.setPlaceholderText("e.g. 192.168.0.12 or 172.20.10.2")
    quality_box = QComboBox()
    quality_box.setFixedWidth(460)
    quality_labels = {
        "balanced": "Balanced (recommended)",
        "low-power": "Low power",
        "high": "High quality",
    }
    for profile in QUALITY_PROFILES:
        quality_box.addItem(quality_labels[profile], profile)
    selected_quality = initial_quality or store.load_quality() or "balanced"
    selected_index = quality_box.findData(selected_quality)
    if selected_index >= 0:
        quality_box.setCurrentIndex(selected_index)
    form.addRow("Source", mode_box)
    form.addRow("Saved stream", source_box)
    form.addRow("RTSP address", url_field)
    form.addRow("Mac LAN IP (fallback)", publish_host_field)
    form.addRow("Quality", quality_box)
    layout.addLayout(form)

    source_help = QLabel()
    source_help.setWordWrap(True)
    source_help.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(source_help)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Open
    )
    layout.addWidget(buttons)

    def select_preset(index: int) -> None:
        url = source_box.itemData(index)
        if isinstance(url, str) and url:
            url_field.setText(url)

    def set_row_visible(field, visible: bool) -> None:
        field.setVisible(visible)
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    def moblin_publish_url() -> str | None:
        host = publish_host_field.text().strip()
        return f"rtmp://{host}:1935/nuri" if is_reachable_lan_ipv4(host) else None

    def update_source_mode(_index: int | None = None) -> None:
        mode = mode_box.currentData()
        uses_local_url = mode in {"local-demo", "local-relay"}
        set_row_visible(source_box, mode == "rtsp")
        set_row_visible(url_field, mode == "rtsp")
        set_row_visible(publish_host_field, mode == "local-relay")
        if uses_local_url:
            url_field.setText(DEFAULT_DEMO_SOURCE_URL)
        if mode == "local-demo":
            source_help.setText("WithNuri starts the included dog-video stream automatically.")
        elif mode == "local-relay":
            publish_url = moblin_publish_url()
            if publish_url:
                source_help.setText(
                    f"In Moblin, publish to {publish_url}. "
                    "WithNuri starts the local relay and waits for the phone."
                )
            else:
                source_help.setText(
                    "Could not detect a reachable Mac LAN IP. Connect the Mac and "
                    "phone to the same Wi-Fi, then reopen this window. Enter an IP "
                    "above only if you already know it."
                )
        else:
            source_help.setText("Enter an RTSP address or choose a saved stream.")
        if not uses_local_url and url_field.text() == DEFAULT_DEMO_SOURCE_URL:
            saved_url = source_box.currentData()
            url_field.setText(saved_url if isinstance(saved_url, str) else "")

    if initial_url:
        try:
            initial_is_local = normalize_rtsp_url(initial_url) == DEFAULT_DEMO_SOURCE_URL
        except ValueError:
            initial_is_local = False
        if not initial_is_local:
            mode_box.setCurrentIndex(2)

    def accept() -> None:
        mode = mode_box.currentData()
        if mode == "local-demo":
            dialog.accept()
            return
        if mode == "local-relay":
            if moblin_publish_url() is None:
                QMessageBox.warning(
                    dialog,
                    "No reachable Mac LAN IP",
                    "Connect the Mac and phone to the same Wi-Fi, then reopen this "
                    "window. If you already know the Mac LAN IP, enter it above. "
                    "Do not use 127.0.0.1; the phone cannot reach it.",
                )
                return
            dialog.accept()
            return
        try:
            selected = normalize_rtsp_url(url_field.text())
        except ValueError as exc:
            QMessageBox.warning(dialog, "Invalid stream address", str(exc))
            return
        url_field.setText(selected)
        dialog.accept()

    source_box.currentIndexChanged.connect(select_preset)
    mode_box.currentIndexChanged.connect(update_source_mode)
    publish_host_field.textChanged.connect(update_source_mode)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(accept)
    update_source_mode()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    quality = quality_box.currentData()
    if quality not in QUALITY_PROFILES:
        quality = "balanced"
    selected_mode = mode_box.currentData()
    is_local_demo = selected_mode == "local-demo"
    is_local_relay = selected_mode == "local-relay"
    url = store.remember_selection(
        DEFAULT_DEMO_SOURCE_URL if (is_local_demo or is_local_relay) else url_field.text(),
        quality=quality,
    )
    return OverlayLaunchSelection(
        url=url,
        quality=quality,
        start_local_demo=is_local_demo,
        start_local_relay=is_local_relay,
    )
