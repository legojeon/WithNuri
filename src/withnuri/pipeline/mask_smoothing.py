from dataclasses import dataclass

from PIL import Image, ImageFilter

from withnuri.pipeline.yolo_tracker import PetTrackOverlay


@dataclass
class _MaskState:
    mask: Image.Image
    bbox: tuple[int, int, int, int] | None


class TemporalMaskSmoother:
    """Cleans and temporally stabilizes masks for individual display tracks."""

    def __init__(
        self,
        *,
        temporal_weight: float = 0.25,
        morphology_radius: int = 1,
        max_normalized_shift: float = 1.25,
    ) -> None:
        if not 0 <= temporal_weight < 1:
            raise ValueError("temporal_weight must be between 0 and 1 (exclusive)")
        if morphology_radius < 0:
            raise ValueError("morphology_radius must not be negative")
        if max_normalized_shift <= 0:
            raise ValueError("max_normalized_shift must be positive")
        self._temporal_weight = temporal_weight
        self._morphology_radius = morphology_radius
        self._max_normalized_shift = max_normalized_shift
        self._states: dict[int, _MaskState] = {}

    def smooth(self, track: PetTrackOverlay, mask: Image.Image) -> Image.Image:
        cleaned = _morphology(mask, self._morphology_radius, bbox=track.bbox)
        if track.track_id is None:
            return cleaned
        previous = self._states.get(track.track_id)
        result = cleaned
        if previous is not None and _can_blend(
            previous.bbox,
            track.bbox,
            max_normalized_shift=self._max_normalized_shift,
        ):
            dx, dy = _bbox_offset(previous.bbox, track.bbox)
            translated = _translate(previous.mask, dx, dy)
            result = Image.blend(cleaned, translated, self._temporal_weight)
        self._states[track.track_id] = _MaskState(mask=result, bbox=track.bbox)
        return result

    def retain_only(self, track_ids: set[int]) -> None:
        self._states = {
            track_id: state
            for track_id, state in self._states.items()
            if track_id in track_ids
        }

    def reset(self) -> None:
        """Discard prior masks when frames resume from a new stream session."""
        self._states = {}


def _morphology(
    mask: Image.Image,
    radius: int,
    *,
    bbox: tuple[int, int, int, int] | None,
) -> Image.Image:
    if radius == 0:
        return mask
    if bbox is None:
        return _apply_morphology(mask, radius)
    margin = radius * 2 + 4
    left = max(0, bbox[0] - margin)
    top = max(0, bbox[1] - margin)
    right = min(mask.width, bbox[2] + margin)
    bottom = min(mask.height, bbox[3] + margin)
    if right <= left or bottom <= top:
        return mask
    if (right - left) * (bottom - top) > mask.width * mask.height * 0.35:
        # Pillow morphology is CPU-heavy. For a close-up pet, keep temporal
        # smoothing but skip this cosmetic pass rather than dropping frames.
        return mask
    cleaned = Image.new("L", mask.size, 0)
    cleaned.paste(_apply_morphology(mask.crop((left, top, right, bottom)), radius), (left, top))
    return cleaned


def _apply_morphology(mask: Image.Image, radius: int) -> Image.Image:
    size = radius * 2 + 1
    # Closing fills small holes, then opening removes isolated foreground noise.
    return mask.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size)).filter(
        ImageFilter.MinFilter(size)
    ).filter(ImageFilter.MaxFilter(size))


def _can_blend(
    previous: tuple[int, int, int, int] | None,
    current: tuple[int, int, int, int] | None,
    *,
    max_normalized_shift: float,
) -> bool:
    if previous is None or current is None:
        return False
    dx, dy = _bbox_offset(previous, current)
    distance = (dx * dx + dy * dy) ** 0.5
    scale = max((_bbox_diagonal(previous) + _bbox_diagonal(current)) / 2, 1.0)
    return distance / scale <= max_normalized_shift


def _translate(mask: Image.Image, dx: float, dy: float) -> Image.Image:
    return mask.transform(
        mask.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def _bbox_offset(
    previous: tuple[int, int, int, int] | None,
    current: tuple[int, int, int, int] | None,
) -> tuple[float, float]:
    if previous is None or current is None:
        return 0.0, 0.0
    previous_center_x = (previous[0] + previous[2]) / 2
    previous_center_y = (previous[1] + previous[3]) / 2
    current_center_x = (current[0] + current[2]) / 2
    current_center_y = (current[1] + current[3]) / 2
    return current_center_x - previous_center_x, current_center_y - previous_center_y


def _bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
