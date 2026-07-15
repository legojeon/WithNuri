from threading import Lock
from typing import Generic, TypeVar


FrameT = TypeVar("FrameT")


class LatestFrameQueue(Generic[FrameT]):
    def __init__(self) -> None:
        self._frame: FrameT | None = None
        self._lock = Lock()
        self._dropped_frame_count = 0

    def put(self, frame: FrameT) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped_frame_count += 1
            self._frame = frame

    def latest(self) -> FrameT | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    @property
    def dropped_frame_count(self) -> int:
        with self._lock:
            return self._dropped_frame_count
