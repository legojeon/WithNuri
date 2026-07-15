from dataclasses import dataclass

from PIL import Image

from withnuri.pipeline.frames import ProcessedFrame
from withnuri.pipeline.yolo_tracker import PetTrackOverlay


@dataclass(frozen=True)
class CropRegion:
    left: float
    top: float
    right: float
    bottom: float


class PetCropController:
    """Keeps a padded, aspect-correct crop around all currently tracked pets."""

    def __init__(
        self,
        *,
        output_width: int,
        output_height: int,
        padding_ratio: float = 0.35,
        smoothing: float = 0.35,
    ) -> None:
        if output_width <= 0 or output_height <= 0:
            raise ValueError("crop output dimensions must be positive")
        if padding_ratio < 0:
            raise ValueError("crop padding_ratio must not be negative")
        if not 0 < smoothing <= 1:
            raise ValueError("crop smoothing must be between 0 (exclusive) and 1")
        self._output_width = output_width
        self._output_height = output_height
        self._padding_ratio = padding_ratio
        self._smoothing = smoothing
        self._region: CropRegion | None = None

    @property
    def region(self) -> CropRegion | None:
        return self._region

    def reset(self) -> None:
        """Discard the old crop when frames resume from a new stream session."""
        self._region = None

    def set_output_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("crop output dimensions must be positive")
        self._output_width = width
        self._output_height = height

    def crop(
        self, frame: ProcessedFrame, tracks: list[PetTrackOverlay]
    ) -> ProcessedFrame:
        target = _target_region(
            tracks,
            frame_width=frame.width,
            frame_height=frame.height,
            aspect_ratio=self._output_width / self._output_height,
            padding_ratio=self._padding_ratio,
        )
        if target is not None:
            self._region = _smooth_region(self._region, target, self._smoothing)
        if self._region is None:
            return _resize(frame, self._output_width, self._output_height)
        return _crop_and_resize(
            frame,
            self._region,
            output_width=self._output_width,
            output_height=self._output_height,
        )


def _target_region(
    tracks: list[PetTrackOverlay],
    *,
    frame_width: int,
    frame_height: int,
    aspect_ratio: float,
    padding_ratio: float,
) -> CropRegion | None:
    boxes = [track.bbox for track in tracks if track.bbox is not None]
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    width = max(1, right - left)
    height = max(1, bottom - top)
    padding = max(width, height) * padding_ratio
    target_width = width + padding * 2
    target_height = height + padding * 2
    if target_width / target_height < aspect_ratio:
        target_width = target_height * aspect_ratio
    else:
        target_height = target_width / aspect_ratio
    return _fit_region(
        center_x=(left + right) / 2,
        center_y=(top + bottom) / 2,
        width=min(target_width, frame_width),
        height=min(target_height, frame_height),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _smooth_region(
    previous: CropRegion | None, target: CropRegion, smoothing: float
) -> CropRegion:
    if previous is None:
        return target
    return CropRegion(
        left=_lerp(previous.left, target.left, smoothing),
        top=_lerp(previous.top, target.top, smoothing),
        right=_lerp(previous.right, target.right, smoothing),
        bottom=_lerp(previous.bottom, target.bottom, smoothing),
    )


def _fit_region(
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> CropRegion:
    left = min(max(0.0, center_x - width / 2), frame_width - width)
    top = min(max(0.0, center_y - height / 2), frame_height - height)
    return CropRegion(left=left, top=top, right=left + width, bottom=top + height)


def _crop_and_resize(
    frame: ProcessedFrame,
    region: CropRegion,
    *,
    output_width: int,
    output_height: int,
) -> ProcessedFrame:
    image = Image.frombytes("RGBA", (frame.width, frame.height), frame.data)
    cropped = image.crop((region.left, region.top, region.right, region.bottom))
    resized = cropped.resize((output_width, output_height), Image.Resampling.BILINEAR)
    return ProcessedFrame(
        width=output_width,
        height=output_height,
        data=resized.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )


def _resize(frame: ProcessedFrame, width: int, height: int) -> ProcessedFrame:
    image = Image.frombytes("RGBA", (frame.width, frame.height), frame.data)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return ProcessedFrame(
        width=width,
        height=height,
        data=resized.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )


def _lerp(first: float, second: float, ratio: float) -> float:
    return first + (second - first) * ratio
