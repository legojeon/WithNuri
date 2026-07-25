"""Lifecycle management for WithNuri's local relay and demo streams."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import time

from withnuri.runtime import bundled_resource_root, default_ffmpeg_binary


class LocalDemoUnavailable(RuntimeError):
    """Raised when the local MediaMTX/MP4 demo cannot be started."""


@dataclass(frozen=True)
class LocalDemoAssets:
    mediamtx_binary: Path
    mediamtx_config: Path
    video_path: Path


class LocalRelaySession:
    """Starts MediaMTX for a phone publisher and stops only that relay."""

    def __init__(
        self,
        *,
        assets: LocalDemoAssets | None = None,
        popen=subprocess.Popen,
        wait_for_port=None,
    ) -> None:
        self._assets = assets or default_local_demo_assets()
        self._popen = popen
        self._wait_for_port = wait_for_port or wait_for_tcp_port
        self._mediamtx_process = None

    def start(self) -> None:
        self._validate_relay_assets()
        try:
            self._mediamtx_process = self._popen(
                [str(self._assets.mediamtx_binary), str(self._assets.mediamtx_config)],
                cwd=self._assets.mediamtx_config.parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise LocalDemoUnavailable(f"Could not start MediaMTX: {exc}") from exc

        if not self._wait_for_port("127.0.0.1", 8554, timeout_seconds=5):
            self.stop()
            raise LocalDemoUnavailable(
                "MediaMTX did not open its RTSP port. Another relay may already be running."
            )

    def stop(self) -> None:
        _stop_process(self._mediamtx_process)
        self._mediamtx_process = None

    def _validate_relay_assets(self) -> None:
        missing = [
            path
            for path in (
                self._assets.mediamtx_binary,
                self._assets.mediamtx_config,
            )
            if not path.is_file()
        ]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise LocalDemoUnavailable(f"Local relay assets are missing: {formatted}")


class LocalDemoSession(LocalRelaySession):
    """Starts a local relay and looping dog video, then reliably stops both."""

    def __init__(
        self,
        *,
        assets: LocalDemoAssets | None = None,
        ffmpeg_binary: str | None = None,
        popen=subprocess.Popen,
        wait_for_port=None,
    ) -> None:
        super().__init__(assets=assets, popen=popen, wait_for_port=wait_for_port)
        self._ffmpeg_binary = ffmpeg_binary or default_ffmpeg_binary()
        self._publisher_process = None

    def start(self) -> None:
        self._validate_demo_assets()
        super().start()
        try:
            self._publisher_process = self._popen(
                build_looping_mp4_command(
                    self._assets.video_path, ffmpeg_binary=self._ffmpeg_binary
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self.stop()
            raise LocalDemoUnavailable(
                "ffmpeg was not found. Install it or include it in the packaged app."
            ) from exc

    def stop(self) -> None:
        _stop_process(self._publisher_process)
        self._publisher_process = None
        super().stop()

    def _validate_demo_assets(self) -> None:
        if not self._assets.video_path.is_file():
            raise LocalDemoUnavailable(
                f"Local demo assets are missing: {self._assets.video_path}"
            )


def default_local_demo_assets(*, project_root: Path | None = None) -> LocalDemoAssets:
    binary_name = "mediamtx.exe" if os.name == "nt" else "mediamtx"
    bundled_root = bundled_resource_root()
    if bundled_root is not None:
        media_root = bundled_root / "assets" / "mediamtx"
        return LocalDemoAssets(
            mediamtx_binary=media_root / binary_name,
            mediamtx_config=media_root / "mediamtx.yml",
            video_path=bundled_root / "assets" / "dogcam.mp4",
        )
    root = project_root or Path(__file__).resolve().parents[3]
    media_root = root / "WithNuri-tools" / "mediamtx"
    return LocalDemoAssets(
        mediamtx_binary=media_root / binary_name,
        mediamtx_config=media_root / "mediamtx.yml",
        video_path=root / "tests" / "video" / "dogcam.mp4",
    )


def build_looping_mp4_command(video_path: Path, *, ffmpeg_binary: str) -> list[str]:
    video_filter = (
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"
    )
    return [
        ffmpeg_binary,
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-f",
        "flv",
        "rtmp://127.0.0.1:1935/nuri",
    ]


def wait_for_tcp_port(
    host: str, port: int, *, timeout_seconds: float, sleep=time.sleep
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            sleep(0.05)
    return False


def _stop_process(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
