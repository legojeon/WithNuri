"""Local, best-effort storage for user-selected RTSP sources."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from urllib.parse import urlsplit


DEFAULT_DEMO_SOURCE_URL = "rtsp://127.0.0.1:8554/nuri"
QUALITY_PROFILES = ("balanced", "low-power", "high")
_MAX_RECENT_SOURCES = 5


def local_lan_ipv4(*, command_runner=None) -> str | None:
    """Return this machine's likely LAN IPv4 address for phone setup text.

    macOS reports the default route's interface directly, which correctly
    covers Wi-Fi and iPhone Personal Hotspot connections. Other platforms use
    a best-effort UDP route lookup. ``None`` means no reachable phone address
    was available; callers must not substitute localhost for a phone URL.
    """
    runner = command_runner or subprocess.run
    if _sys_platform() == "darwin":
        interface = _macos_default_interface(runner)
        if interface:
            address = _macos_interface_ipv4(interface, runner)
            if is_reachable_lan_ipv4(address):
                return address

        # A default route can be temporarily absent while a hotspot comes up.
        # These are the common Wi-Fi interfaces and provide a useful fallback.
        for interface in ("en0", "en1"):
            address = _macos_interface_ipv4(interface, runner)
            if is_reachable_lan_ipv4(address):
                return address

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("8.8.8.8", 53))
            address = connection.getsockname()[0]
    except OSError:
        return None
    return address if is_reachable_lan_ipv4(address) else None


def _macos_default_interface(runner) -> str | None:
    output = _command_output(["route", "-n", "get", "default"], runner)
    for line in output.splitlines():
        label, separator, value = line.partition(":")
        if separator and label.strip() == "interface" and value.strip():
            return value.strip()
    return None


def _macos_interface_ipv4(interface: str, runner) -> str | None:
    output = _command_output(["ipconfig", "getifaddr", interface], runner)
    address = output.strip()
    if address:
        return address

    output = _command_output(["ifconfig", interface], runner)
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] == "inet":
            return fields[1]
    return None


def _command_output(command: list[str], runner) -> str:
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def is_reachable_lan_ipv4(address: str | None) -> bool:
    try:
        candidate = ipaddress.IPv4Address(address or "")
    except ipaddress.AddressValueError:
        return False
    documentation_networks = (
        ipaddress.IPv4Network("192.0.0.0/24"),
        ipaddress.IPv4Network("192.0.2.0/24"),
        ipaddress.IPv4Network("198.51.100.0/24"),
        ipaddress.IPv4Network("203.0.113.0/24"),
    )
    return not (
        candidate.is_loopback
        or candidate.is_unspecified
        or candidate.is_multicast
        or candidate.is_link_local
        or candidate.is_reserved
        or any(candidate in network for network in documentation_networks)
    )


class StreamSourceStore:
    """Persist recent RTSP URLs without making stream startup depend on disk IO."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_stream_source_path()

    def load_recent(self) -> list[str]:
        payload = self._load_payload()
        urls = payload.get("recent_urls")
        if not isinstance(urls, list):
            return []

        recent: list[str] = []
        for url in urls:
            if not isinstance(url, str):
                continue
            try:
                normalized = normalize_rtsp_url(url)
            except ValueError:
                continue
            if normalized not in recent:
                recent.append(normalized)
        return recent[:_MAX_RECENT_SOURCES]

    def remember(self, url: str) -> str:
        return self.remember_selection(url, quality=self.load_quality())

    def load_quality(self) -> str | None:
        quality = self._load_payload().get("quality")
        return quality if quality in QUALITY_PROFILES else None

    def remember_selection(self, url: str, *, quality: str | None) -> str:
        normalized = normalize_rtsp_url(url)
        recent = [item for item in self.load_recent() if item != normalized]
        recent.insert(0, normalized)
        saved_quality = quality if quality in QUALITY_PROFILES else None
        self._save(recent[:_MAX_RECENT_SOURCES], quality=saved_quality)
        return normalized

    def _load_payload(self) -> dict:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save(self, urls: list[str], *, quality: str | None) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as temporary:
                payload = {"recent_urls": urls}
                if quality is not None:
                    payload["quality"] = quality
                json.dump(payload, temporary, separators=(",", ":"))
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._path)
        except OSError:
            return


def normalize_rtsp_url(value: str) -> str:
    """Validate the source shape early, before ffmpeg reports an opaque error."""
    url = value.strip()
    parsed = urlsplit(url)
    if (
        parsed.scheme.lower() != "rtsp"
        or not parsed.netloc
        or any(character.isspace() for character in url)
    ):
        raise ValueError("Enter a valid RTSP URL, for example rtsp://127.0.0.1:8554/nuri")
    return url


def default_stream_source_path(
    *, home: Path | None = None, environ: dict[str, str] | None = None
) -> Path:
    home = home or Path.home()
    environ = environ or os.environ
    if os.name == "nt":
        root = Path(environ.get("APPDATA", home / "AppData" / "Roaming"))
    elif _sys_platform() == "darwin":
        root = home / "Library" / "Application Support"
    else:
        root = Path(environ.get("XDG_CONFIG_HOME", home / ".config"))
    return root / "WithNuri" / "stream-sources.json"


def _sys_platform() -> str:
    import sys

    return sys.platform
