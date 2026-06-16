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

    for frame in decoder.iter_frames():
        frame_count += 1
        last_frame = frame
        if clock() - started_at >= seconds:
            break

    if frame_count == 0:
        return StreamCheckResult(frame_count=0, observed_fps=0.0, last_frame=None)
    return StreamCheckResult(
        frame_count=frame_count,
        observed_fps=round(frame_count / seconds, 2),
        last_frame=last_frame,
    )
