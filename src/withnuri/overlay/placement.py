"""Persistent, screen-safe placement for the desktop overlay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class OverlayPlacement:
    x: int
    y: int
    width: int
    height: int


class OverlayPlacementStore:
    """Best-effort local store; bad or unavailable settings never block overlay."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_overlay_placement_path()

    def load(self) -> OverlayPlacement | None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            placement = OverlayPlacement(
                x=int(payload["x"]),
                y=int(payload["y"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return placement if placement.width > 0 and placement.height > 0 else None

    def save(self, placement: OverlayPlacement) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as temporary:
                json.dump(asdict(placement), temporary, separators=(",", ":"))
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._path)
        except OSError:
            return

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            return


def default_overlay_placement_path(
    *, home: Path | None = None, environ: dict[str, str] | None = None
) -> Path:
    home = home or Path.home()
    environ = environ or os.environ
    if os.name == "nt":
        root = Path(environ.get("APPDATA", home / "AppData" / "Roaming"))
    elif sys_platform() == "darwin":
        root = home / "Library" / "Application Support"
    else:
        root = Path(environ.get("XDG_CONFIG_HOME", home / ".config"))
    return root / "WithNuri" / "overlay-placement.json"


def sys_platform() -> str:
    import sys

    return sys.platform
