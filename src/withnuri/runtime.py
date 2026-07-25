"""Paths for assets in development and in a frozen macOS application."""

from __future__ import annotations

from pathlib import Path
import sys


def bundled_resource_root() -> Path | None:
    root = getattr(sys, "_MEIPASS", None)
    return Path(root) if root else None


def resource_path(relative_path: str | Path) -> Path:
    """Return a project asset path in development or inside a frozen app."""
    root = bundled_resource_root()
    if root is None:
        root = Path(__file__).resolve().parents[2]
    return root / relative_path


def default_ffmpeg_binary() -> str:
    root = bundled_resource_root()
    bundled = root / "bin" / "ffmpeg" if root is not None else None
    return str(bundled) if bundled is not None and bundled.is_file() else "ffmpeg"


def default_yolo_model_path() -> str:
    root = bundled_resource_root()
    bundled = root / "yolo11n-seg.pt" if root is not None else None
    return str(bundled) if bundled is not None and bundled.is_file() else "yolo11n-seg.pt"
