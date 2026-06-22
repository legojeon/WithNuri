from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from withnuri.pipeline.frames import RawFrame


class FrameDecoder(Protocol):
    def iter_frames(self, *, max_frames: int | None = None): ...


@dataclass(frozen=True)
class StreamCheckResult:
    frame_count: int
    observed_fps: float
    last_frame: RawFrame | None


def check_stream(
    decoder: FrameDecoder, *, seconds: float, clock=monotonic
) -> StreamCheckResult:
    started_at = clock()
    frame_count = 0
    last_frame = None
    elapsed = 0.0

    for frame in decoder.iter_frames():
        frame_count += 1
        last_frame = frame
        elapsed = clock() - started_at
        if elapsed >= seconds:
            break

    if frame_count == 0 or elapsed <= 0:
        return StreamCheckResult(
            frame_count=frame_count, observed_fps=0.0, last_frame=last_frame
        )
    return StreamCheckResult(
        frame_count=frame_count,
        observed_fps=round(frame_count / elapsed, 2),
        last_frame=last_frame,
    )
