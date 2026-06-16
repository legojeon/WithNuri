from dataclasses import dataclass


_CHANNEL_COUNTS = {
    "rgb24": 3,
}


@dataclass(frozen=True)
class RawFrame:
    width: int
    height: int
    pixel_format: str
    data: bytes
    timestamp_seconds: float

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        if self.pixel_format not in _CHANNEL_COUNTS:
            raise ValueError(f"unsupported pixel format: {self.pixel_format}")
        if len(self.data) != self.width * self.height * self.channel_count:
            raise ValueError(
                "frame payload size does not match dimensions and pixel format"
            )

    @property
    def channel_count(self) -> int:
        return _CHANNEL_COUNTS[self.pixel_format]

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class ProcessedFrame:
    width: int
    height: int
    data: bytes
    timestamp_seconds: float
    pixel_format: str = "rgba"

    def __post_init__(self) -> None:
        _validate_dimensions(self.width, self.height)
        if self.pixel_format != "rgba":
            raise ValueError("processed frames must be rgba")
        if len(self.data) != self.width * self.height * 4:
            raise ValueError(
                "processed frame payload size does not match RGBA dimensions"
            )

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class MaskFrame:
    width: int
    height: int
    data: bytes
    timestamp_seconds: float
    pixel_format: str = "gray8"

    def __post_init__(self) -> None:
        _validate_single_channel_frame(
            width=self.width,
            height=self.height,
            data=self.data,
            expected_pixel_format="gray8",
            actual_pixel_format=self.pixel_format,
            payload_name="mask",
        )

    @property
    def byte_size(self) -> int:
        return len(self.data)


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")


def _validate_single_channel_frame(
    *,
    width: int,
    height: int,
    data: bytes,
    expected_pixel_format: str,
    actual_pixel_format: str,
    payload_name: str,
) -> None:
    _validate_dimensions(width, height)
    if actual_pixel_format != expected_pixel_format:
        raise ValueError(f"{payload_name} frames must be {expected_pixel_format}")
    if len(data) != width * height:
        raise ValueError(f"{payload_name} payload size does not match dimensions")
