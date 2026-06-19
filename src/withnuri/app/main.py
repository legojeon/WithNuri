import argparse
from collections.abc import Sequence
import subprocess
import sys
import time

from withnuri.pipeline.yolo_tracker import (
    YoloPetTracker,
    YoloTrackingDependencyMissing,
)
from withnuri.streaming.decoder import DecodeToolMissing, FfmpegFrameDecoder
from withnuri.streaming.phone_profiles import MoblinRtmpProfile
from withnuri.streaming.probe import ProbeToolMissing, probe_stream
from withnuri.streaming.stream_check import check_stream
from withnuri.ui.debug_window import QtDebugWindow, run_tracking_debug_preview
from withnuri.ui.windowing import PreviewWindowUnavailable


TRACKER_NAME = "yolo11n-seg + ByteTrack"


def main(
    argv: Sequence[str] | None = None,
    *,
    runner=subprocess.run,
    decoder_factory=FfmpegFrameDecoder,
    yolo_tracker_factory=YoloPetTracker,
    debug_window_factory=QtDebugWindow,
    debug_runner=run_tracking_debug_preview,
    clock=time.monotonic,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "moblin-rtmp":
        return _run_moblin_rtmp(args)
    if args.command == "probe":
        return _run_probe(args, runner=runner)
    if args.command == "grab-frame":
        return _run_grab_frame(args, decoder_factory=decoder_factory)
    if args.command == "stream-check":
        return _run_stream_check(args, decoder_factory=decoder_factory, clock=clock)
    if args.command == "tracking-debug":
        return _run_tracking_debug(
            args,
            decoder_factory=decoder_factory,
            yolo_tracker_factory=yolo_tracker_factory,
            debug_window_factory=debug_window_factory,
            debug_runner=debug_runner,
        )

    parser.error(f"unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="withnuri")
    subparsers = parser.add_subparsers(dest="command", required=True)

    moblin_parser = subparsers.add_parser(
        "moblin-rtmp", help="Print Moblin RTMP setup values."
    )
    moblin_parser.add_argument("--host", required=True)
    moblin_parser.add_argument("--stream-name", required=True)

    probe_parser = subparsers.add_parser("probe", help="Probe a readable stream URL.")
    probe_parser.add_argument("url")

    grab_parser = subparsers.add_parser(
        "grab-frame", help="Read one frame from a stream URL."
    )
    grab_parser.add_argument("url")
    _add_frame_size_args(grab_parser)

    stream_check_parser = subparsers.add_parser(
        "stream-check", help="Read frames for a short period."
    )
    stream_check_parser.add_argument("url")
    stream_check_parser.add_argument("--seconds", type=float, default=3)
    _add_frame_size_args(stream_check_parser)

    tracking_parser = subparsers.add_parser(
        "tracking-debug",
        help="Show a camera debug window with YOLO masks and ByteTrack IDs.",
    )
    tracking_parser.add_argument("url")
    _add_frame_size_args(tracking_parser)
    tracking_parser.add_argument("--display-width", type=int)
    tracking_parser.add_argument("--display-height", type=int)
    tracking_parser.add_argument("--decode-fps", type=float)
    tracking_parser.add_argument("--max-frames", type=int)
    tracking_parser.add_argument("--confidence", type=float, default=0.25)
    tracking_parser.add_argument("--image-size", type=int, default=640)
    tracking_parser.add_argument("--mask-cache-frames", type=int, default=8)
    return parser


def _add_frame_size_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)


def _run_moblin_rtmp(args) -> int:
    profile = MoblinRtmpProfile(host=args.host, stream_name=args.stream_name)
    print(f"Publish URL: {profile.publish_url}")
    print(f"WithNuri read URL: {profile.read_url}")
    print()
    print("Moblin setup:")
    for index, instruction in enumerate(profile.instructions, start=1):
        print(f"{index}. {instruction}")
    return 0


def _run_probe(args, *, runner) -> int:
    try:
        result = probe_stream(args.url, runner=runner)
    except ProbeToolMissing:
        print(
            "ffprobe was not found. Install ffmpeg/ffprobe or configure a bundled probe tool.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"Stream probe failed: {exc}", file=sys.stderr)
        return 1

    frame_rate = "unknown" if result.frame_rate is None else f"{result.frame_rate:g}"
    print(f"Stream reachable: {result.url}")
    if result.video_codec:
        print(
            f"Video: {result.video_codec} {result.width}x{result.height} {frame_rate} fps"
        )
    else:
        print("Video: none")
    print(f"Audio: {result.audio_codec or 'none'}")
    return 0


def _run_grab_frame(args, *, decoder_factory) -> int:
    decoder = decoder_factory(args.url, args.width, args.height)
    try:
        frame = decoder.read_one_frame()
    except DecodeToolMissing:
        print(
            "ffmpeg was not found. Install ffmpeg or configure a bundled decode tool.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"Frame capture failed: {exc}", file=sys.stderr)
        return 1

    print(f"Frame captured: {args.url}")
    print(f"Format: {frame.pixel_format} {frame.width}x{frame.height}")
    print(f"Bytes: {frame.byte_size}")
    return 0


def _run_stream_check(args, *, decoder_factory, clock) -> int:
    decoder = decoder_factory(args.url, args.width, args.height)
    try:
        result = check_stream(decoder, seconds=args.seconds, clock=clock)
    except DecodeToolMissing:
        print(
            "ffmpeg was not found. Install ffmpeg or configure a bundled decode tool.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"Stream check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Stream checked: {args.url}")
    print(f"Frames read: {result.frame_count}")
    print(f"Observed FPS: {result.observed_fps:g}")
    if result.last_frame is None:
        print("Last frame: none")
    else:
        frame = result.last_frame
        print(
            f"Last frame: {frame.pixel_format} {frame.width}x{frame.height}, {frame.byte_size} bytes"
        )
    return 0


def _run_tracking_debug(
    args,
    *,
    decoder_factory,
    yolo_tracker_factory,
    debug_window_factory,
    debug_runner,
) -> int:
    decoder = decoder_factory(
        args.url,
        args.width,
        args.height,
        output_fps=args.decode_fps,
        reraise_interrupts_on_cleanup=True,
    )
    try:
        tracker = yolo_tracker_factory(
            confidence=args.confidence,
            image_size=args.image_size,
            cache_frames=args.mask_cache_frames,
        )
        if (args.display_width is None) != (args.display_height is None):
            raise RuntimeError(
                "--display-width and --display-height must be used together"
            )
        window = debug_window_factory(
            title="WithNuri Debug Overlay",
            always_on_top=True,
            display_width=args.display_width,
            display_height=args.display_height,
        )
        result = debug_runner(
            decoder, tracker=tracker, window=window, max_frames=args.max_frames
        )
    except DecodeToolMissing:
        print(
            "ffmpeg was not found. Install ffmpeg or configure a bundled decode tool.",
            file=sys.stderr,
        )
        return 2
    except YoloTrackingDependencyMissing as exc:
        print(f"YOLO tracking is unavailable: {exc}", file=sys.stderr)
        return 2
    except PreviewWindowUnavailable as exc:
        print(f"Tracking debug window is unavailable: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Tracking debug failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Tracking debug interrupted.")
        return 130

    if result.interrupted:
        _print_tracking_debug_result("interrupted", args.url, result)
        return 130
    _print_tracking_debug_result("closed", args.url, result)
    return 0


def _print_tracking_debug_result(status: str, url: str, result) -> None:
    print(f"Tracking debug {status}: {url}")
    print(f"Tracker: {TRACKER_NAME}")
    print(f"Frames rendered: {result.frames_rendered}")
    if status == "interrupted":
        print(
            f"Visible pet frames: {result.visible_frame_count}/{result.frames_rendered}"
        )
        print(f"Last frame visible: {'yes' if result.last_frame_visible else 'no'}")
