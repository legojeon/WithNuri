from PIL import Image, ImageDraw

from withnuri.pipeline.frames import ProcessedFrame, RawFrame
from withnuri.pipeline.yolo_tracker import PetTrackingFrame


YOLO_GREEN = (0, 255, 0, 255)
MASK_GREEN = (0, 255, 0, 80)
TRACK_BLUE = (0, 120, 255, 255)


def render_tracking_debug_overlay(
    frame: RawFrame, tracking: PetTrackingFrame
) -> ProcessedFrame:
    if frame.pixel_format != "rgb24":
        raise ValueError("debug overlay expects an rgb24 RawFrame")
    if frame.width != tracking.width or frame.height != tracking.height:
        raise ValueError("frame and tracking result must have the same dimensions")

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.data).convert(
        "RGBA"
    )
    draw = ImageDraw.Draw(image)

    for track in tracking.tracks:
        _draw_mask(image, track.mask.data, width=frame.width, height=frame.height)
        if track.yolo_bbox is not None:
            draw.rectangle(track.yolo_bbox, outline=YOLO_GREEN, width=1)
        if track.track_bbox is not None:
            draw.rectangle(track.track_bbox, outline=TRACK_BLUE, width=1)
            if track.track_id is not None:
                x1, y1, _, _ = track.track_bbox
                draw.text(
                    (x1, max(0, y1 - 12)), f"ID {track.track_id}", fill=TRACK_BLUE
                )

    return ProcessedFrame(
        width=frame.width,
        height=frame.height,
        data=image.tobytes(),
        timestamp_seconds=frame.timestamp_seconds,
    )


def _draw_mask(
    image: Image.Image, mask_data: bytes, *, width: int, height: int
) -> None:
    mask = Image.frombytes("L", (width, height), mask_data)
    overlay = Image.new("RGBA", (width, height), MASK_GREEN)
    image.alpha_composite(
        Image.composite(overlay, Image.new("RGBA", (width, height), (0, 0, 0, 0)), mask)
    )
