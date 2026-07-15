from dataclasses import dataclass

from withnuri.pipeline.yolo_tracker import PetTrackOverlay, PetTrackingFrame


@dataclass(frozen=True)
class DisplayStabilizationMetrics:
    suppressed_detection_frames: int
    confirmed_tracks: int
    reassociation_events: int
    grace_retained_frames: int
    expired_tracks: int


@dataclass
class _DisplayTrack:
    display_track_id: int | None
    source_track_id: int | None
    last_track: PetTrackOverlay
    detection_streak: int = 1
    confirmed: bool = False
    missed_frames: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class PetDisplayStabilizer:
    """Turns noisy frame-level tracks into stable overlay display tracks.

    A newly observed pet is confirmed only after consecutive observations. Once
    confirmed, a short missing interval retains its state for re-association.
    If a tracker assigns a new ID at a nearby bounding box, it is treated as
    the same display pet instead of making the overlay disappear and reappear.
    """

    def __init__(
        self,
        *,
        confirm_frames: int = 2,
        grace_frames: int = 8,
        reassociation_iou: float = 0.2,
        reassociation_center_distance: float = 1.25,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if grace_frames < 0:
            raise ValueError("grace_frames must not be negative")
        if not 0 <= reassociation_iou <= 1:
            raise ValueError("reassociation_iou must be between 0 and 1")
        if reassociation_center_distance <= 0:
            raise ValueError("reassociation_center_distance must be positive")
        self._confirm_frames = confirm_frames
        self._grace_frames = grace_frames
        self._reassociation_iou = reassociation_iou
        self._reassociation_center_distance = reassociation_center_distance
        self._states: list[_DisplayTrack] = []
        self._suppressed_detection_frames = 0
        self._confirmed_tracks = 0
        self._reassociation_events = 0
        self._grace_retained_frames = 0
        self._expired_tracks = 0

    def metrics(self) -> DisplayStabilizationMetrics:
        return DisplayStabilizationMetrics(
            suppressed_detection_frames=self._suppressed_detection_frames,
            confirmed_tracks=self._confirmed_tracks,
            reassociation_events=self._reassociation_events,
            grace_retained_frames=self._grace_retained_frames,
            expired_tracks=self._expired_tracks,
        )

    def reset(self) -> None:
        """Discard display identities when the upstream stream reconnects."""
        self._states = []

    def stabilize(self, tracking: PetTrackingFrame) -> PetTrackingFrame:
        detected_tracks = [track for track in tracking.tracks if track.detected]
        matched_state_indexes: set[int] = set()

        for track in detected_tracks:
            match = self._find_state(track, matched_state_indexes)
            if match is None:
                confirmed = self._confirm_frames == 1
                self._states.append(
                    _DisplayTrack(
                        display_track_id=track.track_id,
                        source_track_id=track.track_id,
                        last_track=track,
                        confirmed=confirmed,
                    )
                )
                if confirmed:
                    self._confirmed_tracks += 1
                else:
                    self._suppressed_detection_frames += 1
                matched_state_indexes.add(len(self._states) - 1)
                continue

            state_index, reassociated = match
            state = self._states[state_index]
            if reassociated:
                self._reassociation_events += 1
            state.velocity_x, state.velocity_y = _bbox_velocity(
                state.last_track.bbox, track.bbox
            )
            state.source_track_id = track.track_id
            state.last_track = track
            state.missed_frames = 0
            if not state.confirmed:
                state.detection_streak += 1
                state.confirmed = state.detection_streak >= self._confirm_frames
                if state.confirmed:
                    self._confirmed_tracks += 1
                    self._retire_competing_grace_states(
                        confirmed_index=state_index,
                        matched_state_indexes=matched_state_indexes,
                    )
                else:
                    self._suppressed_detection_frames += 1
            matched_state_indexes.add(state_index)

        display_tracks: list[PetTrackOverlay] = []
        retained_states: list[_DisplayTrack] = []
        for index, state in enumerate(self._states):
            if index not in matched_state_indexes:
                state.missed_frames += 1
            if state.missed_frames > self._grace_frames:
                self._expired_tracks += 1
                continue
            retained_states.append(state)
            if not state.confirmed:
                continue
            if index in matched_state_indexes:
                display_tracks.append(
                    PetTrackOverlay(
                        track_id=state.display_track_id,
                        bbox=state.last_track.bbox,
                        mask=state.last_track.mask,
                        detected=True,
                    )
                )
            else:
                self._grace_retained_frames += 1
                display_tracks.append(
                    PetTrackOverlay(
                        track_id=state.display_track_id,
                        bbox=state.last_track.bbox,
                        mask=state.last_track.mask,
                        detected=False,
                        cache_age=state.missed_frames,
                    )
                )
        self._states = retained_states
        return PetTrackingFrame(
            width=tracking.width,
            height=tracking.height,
            timestamp_seconds=tracking.timestamp_seconds,
            tracks=display_tracks,
        )

    def _retire_competing_grace_states(
        self, *, confirmed_index: int, matched_state_indexes: set[int]
    ) -> None:
        confirmed = self._states[confirmed_index]
        for index, state in enumerate(self._states):
            if index == confirmed_index or index in matched_state_indexes:
                continue
            if not state.confirmed:
                continue
            distance = _normalized_center_distance(
                confirmed.last_track.bbox,
                _predicted_bbox(state),
            )
            if distance <= self._reassociation_center_distance:
                state.missed_frames = self._grace_frames + 1

    def _find_state(
        self, track: PetTrackOverlay, matched_state_indexes: set[int]
    ) -> tuple[int, bool] | None:
        for index, state in enumerate(self._states):
            if index in matched_state_indexes:
                continue
            if track.track_id is not None and track.track_id == state.source_track_id:
                return index, False

        best_index = None
        best_score = -1.0
        for index, state in enumerate(self._states):
            if index in matched_state_indexes:
                continue
            predicted_bbox = _predicted_bbox(state)
            iou = _bbox_iou(track.bbox, predicted_bbox)
            center_distance = _normalized_center_distance(track.bbox, predicted_bbox)
            max_center_distance = self._reassociation_center_distance * (
                1 + state.missed_frames * 0.25
            )
            if iou < self._reassociation_iou and center_distance > max_center_distance:
                continue
            # Prefer overlap matches. For non-overlapping fast movement, choose
            # the closest predicted center instead of fragmenting the display ID.
            score = 1 + iou if iou >= self._reassociation_iou else 1 - (
                center_distance / max_center_distance
            )
            if score > best_score:
                best_index = index
                best_score = score
        return (best_index, True) if best_index is not None else None


def _bbox_iou(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _bbox_velocity(
    previous: tuple[int, int, int, int] | None,
    current: tuple[int, int, int, int] | None,
) -> tuple[float, float]:
    if previous is None or current is None:
        return 0.0, 0.0
    previous_x, previous_y = _bbox_center(previous)
    current_x, current_y = _bbox_center(current)
    return current_x - previous_x, current_y - previous_y


def _predicted_bbox(state: _DisplayTrack) -> tuple[float, float, float, float] | None:
    bbox = state.last_track.bbox
    if bbox is None:
        return None
    advance = state.missed_frames + 1
    dx = state.velocity_x * advance
    dy = state.velocity_y * advance
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


def _normalized_center_distance(
    first: tuple[int, int, int, int] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return float("inf")
    first_x, first_y = _bbox_center(first)
    second_x, second_y = _bbox_center(second)
    distance = ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5
    first_diagonal = _bbox_diagonal(first)
    second_diagonal = _bbox_diagonal(second)
    scale = max((first_diagonal + second_diagonal) / 2, 1.0)
    return distance / scale


def _bbox_center(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float]
) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _bbox_diagonal(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float]
) -> float:
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
