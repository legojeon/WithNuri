from PIL import Image, ImageChops, ImageFilter

from withnuri.pipeline.frames import ProcessedFrame, RawFrame
from withnuri.pipeline.yolo_tracker import PetTrackingFrame


def render_pet_matte(
    frame: RawFrame, tracking: PetTrackingFrame, *, feather_radius: float = 1.5
) -> ProcessedFrame:
    if frame.pixel_format != "rgb24":
        raise ValueError("pet matte expects an rgb24 RawFrame")
    if frame.width != tracking.width or frame.height != tracking.height:
        raise ValueError("frame and tracking result must have the same dimensions")

    size = (frame.width, frame.height)
    rgb = Image.frombytes("RGB", size, frame.data)

    alpha = Image.new("L", size, 0)
    for track in tracking.tracks:
        mask = Image.frombytes("L", size, track.mask.data)
        alpha = ImageChops.lighter(alpha, mask)  # union = per-pixel max
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
    scaled = image.point(lambda value: int(value * factor))
    return ProcessedFrame(
        width=frame.width,
        height=frame.height,
        data=scaled.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )
