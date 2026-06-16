from collections.abc import Callable
from dataclasses import dataclass
import json
import subprocess
from urllib.parse import urlparse


ProbeRunner = Callable[..., subprocess.CompletedProcess[str]]


class ProbeToolMissing(RuntimeError):
    """Raised when the configured ffprobe executable cannot be found."""


@dataclass(frozen=True)
class StreamProbeResult:
    url: str
    video_codec: str | None = None
    audio_codec: str | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None


def probe_stream(
    url: str,
    *,
    runner: ProbeRunner = subprocess.run,
    ffprobe_binary: str = "ffprobe",
    timeout_seconds: int = 10,
) -> StreamProbeResult:
    parsed = urlparse(url)
    if parsed.scheme not in {"rtsp", "rtmp"}:
        raise ValueError("stream URL must use rtsp:// or rtmp://")

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        str(timeout_seconds * 1_000_000),
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        url,
    ]
    try:
        completed = runner(
            command, timeout=timeout_seconds, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise ProbeToolMissing(f"{ffprobe_binary} was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timed out after {timeout_seconds} seconds") from exc

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")

    return _parse_probe_output(url, completed.stdout)


def _parse_probe_output(url: str, output: str) -> StreamProbeResult:
    payload = json.loads(output)
    video_stream = None
    audio_stream = None
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    return StreamProbeResult(
        url=url,
        video_codec=video_stream.get("codec_name") if video_stream else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        width=video_stream.get("width") if video_stream else None,
        height=video_stream.get("height") if video_stream else None,
        frame_rate=_parse_frame_rate(video_stream.get("r_frame_rate"))
        if video_stream
        else None,
    )


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(numerator)
    return round(int(numerator) / int(denominator), 2)
