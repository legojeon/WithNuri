from PIL import Image, ImageChops, ImageFilter

from withnuri.pipeline.frames import ProcessedFrame, RawFrame
from withnuri.pipeline.mask_smoothing import TemporalMaskSmoother
from withnuri.pipeline.yolo_tracker import PetTrackingFrame


def render_pet_matte(
    frame: RawFrame,
    tracking: PetTrackingFrame,
    *,
    feather_radius: float = 1.5,
    mask_smoother: TemporalMaskSmoother | None = None,
) -> ProcessedFrame:
    if frame.pixel_format != "rgb24":
        raise ValueError("pet matte expects an rgb24 RawFrame")
    if frame.width != tracking.width or frame.height != tracking.height:
        raise ValueError("frame and tracking result must have the same dimensions")

    size = (frame.width, frame.height)
    rgb = Image.frombytes("RGB", size, frame.data)

    alpha = Image.new("L", size, 0)
    active_track_ids: set[int] = set()
    for track in tracking.tracks:
        # A cached track has no mask for this frame. Compositing its old mask
        # over the latest camera image leaves a cutout at the pet's previous
        # position when it moves quickly. The overlay runner instead holds and
        # fades the last fully rendered frame while a detection is absent.
        if not track.detected:
            continue
        if track.mask.width != frame.width or track.mask.height != frame.height:
            raise ValueError("track mask dimensions must match the frame")
        mask = Image.frombytes("L", size, track.mask.data)
        if mask_smoother is not None:
            mask = mask_smoother.smooth(track, mask)
            if track.track_id is not None:
                active_track_ids.add(track.track_id)
        alpha = ImageChops.lighter(alpha, mask)  # union = per-pixel max
    if mask_smoother is not None:
        mask_smoother.retain_only(active_track_ids)
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather_radius))

    # composite(rgb, black, alpha) == rgb * (alpha/255) == premultiplied RGB.
    premultiplied = Image.composite(rgb, Image.new("RGB", size, 0), alpha)
    rgba = premultiplied.convert("RGBA")
    rgba.putalpha(alpha)

    return ProcessedFrame(
        width=frame.width,
        height=frame.height,
        data=rgba.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )


def scale_processed_alpha(frame: ProcessedFrame, factor: float) -> ProcessedFrame:
    factor = max(0.0, min(1.0, factor))
    image = Image.frombytes("RGBA", (frame.width, frame.height), frame.data)
    scaled = image.point(lambda value: round(value * factor))
    return ProcessedFrame(
        width=frame.width,
        height=frame.height,
        data=scaled.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )
