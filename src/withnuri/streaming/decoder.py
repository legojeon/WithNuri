from collections.abc import Callable
from dataclasses import dataclass, field
import subprocess
import time

from withnuri.pipeline.frames import RawFrame


PopenFactory = Callable[..., subprocess.Popen[bytes]]


class DecodeToolMissing(RuntimeError):
    """Raised when the configured ffmpeg executable cannot be found."""


@dataclass(frozen=True)
class FfmpegFrameDecoder:
    url: str
    width: int = 320
    height: int = 180
    output_fps: float | None = None
    ffmpeg_binary: str = "ffmpeg"
    read_timeout_seconds: int = 10
    reraise_interrupts_on_cleanup: bool = False
    popen: PopenFactory = subprocess.Popen
    now: Callable[[], float] = field(default=time.monotonic)

    def read_one_frame(self) -> RawFrame:
        command = self._command(single_frame=True)
        try:
            process = self.popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise DecodeToolMissing(f"{self.ffmpeg_binary} was not found") from exc

        expected_size = self.width * self.height * 3
        try:
            payload, stderr = process.communicate(timeout=self.read_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise RuntimeError(
                f"timed out after {self.read_timeout_seconds} seconds"
            ) from exc
        if len(payload) != expected_size:
            error = _decode_stderr(stderr)
            raise RuntimeError(error or "incomplete frame payload from ffmpeg")

        return RawFrame(
            width=self.width,
            height=self.height,
            pixel_format="rgb24",
            data=payload,
            timestamp_seconds=self.now(),
        )

    def iter_frames(self, *, max_frames: int | None = None):
        command = self._command(single_frame=False)
        try:
            process = self.popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise DecodeToolMissing(f"{self.ffmpeg_binary} was not found") from exc

        expected_size = self.width * self.height * 3
        frames_read = 0
        assert process.stdout is not None
        try:
            while max_frames is None or frames_read < max_frames:
                payload = process.stdout.read(expected_size)
                if not payload:
                    break
                if len(payload) != expected_size:
                    error = _read_stderr(process)
                    raise RuntimeError(error or "incomplete frame payload from ffmpeg")
                frames_read += 1
                yield RawFrame(
                    width=self.width,
                    height=self.height,
                    pixel_format="rgb24",
                    data=payload,
                    timestamp_seconds=self.now(),
                )
        finally:
            _stop_process(
                process, reraise_interrupts=self.reraise_interrupts_on_cleanup
            )

    def _command(self, *, single_frame: bool) -> list[str]:
        command = [
            self.ffmpeg_binary,
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            str(self.read_timeout_seconds * 1_000_000),
            "-i",
            self.url,
            "-an",
            "-vf",
            self._video_filter(),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        if single_frame:
            frame_filter_index = command.index("-vf")
            command[frame_filter_index:frame_filter_index] = ["-frames:v", "1"]
        return command

    def _video_filter(self) -> str:
        filters = [f"scale={self.width}:{self.height}"]
        if self.output_fps is not None:
            filters.append(f"fps={self.output_fps:g}")
        return ",".join(filters)


def _read_stderr(process) -> str:
    if process.stderr is None:
        return ""
    payload = process.stderr.read()
    if isinstance(payload, bytes):
        return payload.decode(errors="replace").strip()
    return str(payload).strip()


def _decode_stderr(payload) -> str:
    if isinstance(payload, bytes):
        return payload.decode(errors="replace").strip()
    return str(payload or "").strip()


def _stop_process(process, *, reraise_interrupts: bool = False) -> None:
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is not None:
        return
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
        process.wait(timeout=2)
    except KeyboardInterrupt:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
        if reraise_interrupts:
            raise
