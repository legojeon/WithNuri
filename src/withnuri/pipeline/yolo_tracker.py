from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from withnuri.pipeline.frames import MaskFrame, RawFrame


DEFAULT_YOLO_SEGMENT_MODEL = "yolo11n-seg.pt"
TRACKED_CLASS_NAMES = frozenset({"dog", "cat"})
ModelLoader = Callable[[str], Any]

_TRACKER_DISPLAY_NAMES = {
    "bytetrack.yaml": "ByteTrack",
    "botsort.yaml": "BoT-SORT",
}


class YoloTrackingDependencyMissing(RuntimeError):
    """Raised when optional YOLO tracking dependencies are not installed."""


@dataclass(frozen=True)
class PetTrackOverlay:
    track_id: int | None
    bbox: tuple[int, int, int, int] | None
    mask: MaskFrame
    detected: bool
    cache_age: int = 0


@dataclass(frozen=True)
class PetTrackingFrame:
    width: int
    height: int
    timestamp_seconds: float
    tracks: list[PetTrackOverlay]


@dataclass
class YoloPetTracker:
    model_name: str = DEFAULT_YOLO_SEGMENT_MODEL
    confidence: float = 0.25
    image_size: int | tuple[int, int] = 640
    tracker_config: str = "bytetrack.yaml"
    cache_frames: int = 8
    model: Any | None = None
    model_loader: ModelLoader | None = None
    array_module: Any | None = None
    _cache: dict[int, PetTrackOverlay] = field(default_factory=dict, init=False)

    @property
    def display_name(self) -> str:
        model = self.model_name.removesuffix(".pt")
        tracker = _TRACKER_DISPLAY_NAMES.get(self.tracker_config, self.tracker_config)
        return f"{model} + {tracker}"

    def track(self, frame: RawFrame) -> PetTrackingFrame:
        image = rgb24_to_bgr_array(frame, array_module=self._array_module())
        try:
            results = self._model().track(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                retina_masks=True,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"YOLO tracking failed: {exc}") from exc

        detected_tracks = _pet_tracks_from_results(
            results,
            array_module=self._array_module(),
            width=frame.width,
            height=frame.height,
            timestamp_seconds=frame.timestamp_seconds,
        )
        tracks = self._apply_cache(detected_tracks)
        return PetTrackingFrame(
            width=frame.width,
            height=frame.height,
            timestamp_seconds=frame.timestamp_seconds,
            tracks=tracks,
        )

    def _apply_cache(
        self, detected_tracks: list[PetTrackOverlay]
    ) -> list[PetTrackOverlay]:
        tracks = list(detected_tracks)
        current_ids = {
            track.track_id for track in detected_tracks if track.track_id is not None
        }
        next_cache = {
            track.track_id: track
            for track in detected_tracks
            if track.track_id is not None
        }

        for track_id, cached in self._cache.items():
            if track_id in current_ids:
                continue
            cache_age = cached.cache_age + 1
            if cache_age > self.cache_frames:
                continue
            retained = PetTrackOverlay(
                track_id=track_id,
                bbox=cached.bbox,
                mask=cached.mask,
                detected=False,
                cache_age=cache_age,
            )
            tracks.append(retained)
            next_cache[track_id] = retained

        self._cache = next_cache
        return tracks

    def _model(self) -> Any:
        if self.model is None:
            try:
                self.model = self._model_loader()(self.model_name)
            except YoloTrackingDependencyMissing:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load YOLO model {self.model_name}: {exc}"
                ) from exc
        return self.model

    def _model_loader(self) -> ModelLoader:
        if self.model_loader is not None:
            return self.model_loader
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise YoloTrackingDependencyMissing(
                "install WithNuri with the 'ml' extra to use YOLO tracking"
            ) from exc
        return YOLO

    def _array_module(self) -> Any:
        if self.array_module is not None:
            return self.array_module
        try:
            import numpy as np
        except ImportError as exc:
            raise YoloTrackingDependencyMissing(
                "install WithNuri with the 'ml' extra to use YOLO tracking"
            ) from exc
        return np


def rgb24_to_bgr_array(frame: RawFrame, *, array_module: Any):
    if frame.pixel_format != "rgb24":
        raise ValueError("YOLO tracking expects an rgb24 RawFrame")
    rgb = array_module.frombuffer(frame.data, dtype=array_module.uint8).reshape(
        (frame.height, frame.width, 3)
    )
    return rgb[:, :, ::-1].copy()


def _pet_tracks_from_results(
    results: Any,
    *,
    array_module: Any,
    width: int,
    height: int,
    timestamp_seconds: float,
) -> list[PetTrackOverlay]:
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None:
        return []

    names = getattr(result, "names", {})
    class_ids = _to_list(getattr(boxes, "cls", []))
    box_values = _to_list(getattr(boxes, "xyxy", []))
    mask_planes = _to_list(getattr(masks, "data", []))
    track_ids = _track_ids(boxes, count=len(class_ids))

    tracks = []
    for class_id, box, mask_plane, track_id in zip(
        class_ids, box_values, mask_planes, track_ids, strict=False
    ):
        if names.get(int(class_id)) not in TRACKED_CLASS_NAMES:
            continue
        bbox = _bbox_tuple(box)
        tracks.append(
            PetTrackOverlay(
                track_id=None if track_id is None else int(track_id),
                bbox=bbox,
                mask=_mask_from_plane(
                    mask_plane,
                    array_module=array_module,
                    width=width,
                    height=height,
                    timestamp_seconds=timestamp_seconds,
                ),
                detected=True,
                cache_age=0,
            )
        )
    return tracks


def _to_list(value: Any) -> list:
    current = value
    for method_name in ("cpu", "numpy"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(current)


def _track_ids(boxes: Any, *, count: int) -> list[int | None]:
    ids = getattr(boxes, "id", None)
    if ids is None:
        return [None] * count
    return _to_list(ids)


def _bbox_tuple(box: Any) -> tuple[int, int, int, int]:
    values = list(box)
    if len(values) != 4:
        raise RuntimeError("YOLO detection box must have 4 xyxy values")
    return tuple(round(float(value)) for value in values)


def _mask_from_plane(
    mask_plane: Any,
    *,
    array_module: Any,
    width: int,
    height: int,
    timestamp_seconds: float,
) -> MaskFrame:
    plane = array_module.asarray(mask_plane)
    if tuple(plane.shape) != (height, width):
        raise RuntimeError("YOLO detection mask shape does not match frame dimensions")
    data = (plane > 0).astype(array_module.uint8) * 255
    return MaskFrame(
        width=width,
        height=height,
        data=data.tobytes(),
        timestamp_seconds=timestamp_seconds,
    )
