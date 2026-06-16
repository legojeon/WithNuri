from threading import Lock
from typing import Generic, TypeVar


FrameT = TypeVar("FrameT")


class LatestFrameQueue(Generic[FrameT]):
    def __init__(self) -> None:
        self._frame: FrameT | None = None
        self._lock = Lock()

    def put(self, frame: FrameT) -> None:
        with self._lock:
            self._frame = frame

    def latest(self) -> FrameT | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame
