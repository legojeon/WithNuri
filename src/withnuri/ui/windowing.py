from dataclasses import dataclass

from withnuri.pipeline.diagnostics import PipelineDiagnostics
from withnuri.pipeline.frames import ProcessedFrame


@dataclass(frozen=True)
class PreviewResult:
    frames_rendered: int
    interrupted: bool = False
    visible_frame_count: int = 0
    last_frame_visible: bool = False
    diagnostics: PipelineDiagnostics | None = None


class PreviewWindowUnavailable(RuntimeError):
    """Raised when the local Python environment cannot create a preview window."""


def resize_processed_frame(
    frame: ProcessedFrame, *, width: int, height: int
) -> ProcessedFrame:
    if width <= 0 or height <= 0:
        raise ValueError("display dimensions must be positive")
    if frame.width == width and frame.height == height:
        return frame
    from PIL import Image

    image = Image.frombytes("RGBA", (frame.width, frame.height), frame.data)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return ProcessedFrame(
        width=width,
        height=height,
        data=resized.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )
