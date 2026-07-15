"""Local, best-effort storage for user-selected RTSP sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit


DEFAULT_DEMO_SOURCE_URL = "rtsp://127.0.0.1:8554/nuri"
_MAX_RECENT_SOURCES = 5


class StreamSourceStore:
    """Persist recent RTSP URLs without making stream startup depend on disk IO."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_stream_source_path()

    def load_recent(self) -> list[str]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            urls = payload["recent_urls"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return []
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
        normalized = normalize_rtsp_url(url)
        recent = [item for item in self.load_recent() if item != normalized]
        recent.insert(0, normalized)
        self._save(recent[:_MAX_RECENT_SOURCES])
        return normalized

    def _save(self, urls: list[str]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as temporary:
                json.dump({"recent_urls": urls}, temporary, separators=(",", ":"))
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
