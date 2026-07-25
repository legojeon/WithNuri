import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

from withnuri.pipeline.yolo_tracker import (
    YoloPetTracker,
    YoloTrackingDependencyMissing,
)
from withnuri.streaming.decoder import DecodeToolMissing, FfmpegFrameDecoder
from withnuri.streaming.local_demo import (
    LocalDemoSession,
    LocalDemoUnavailable,
    LocalRelaySession,
)
from withnuri.streaming.phone_profiles import MoblinRtmpProfile
from withnuri.streaming.probe import ProbeToolMissing, probe_stream
from withnuri.streaming.stream_check import check_stream
from withnuri.ui.source_dialog import OverlayLaunchSelection, choose_overlay_source
from withnuri.overlay.window import (
    FadeController,
    OverlayWindow,
    TemporalMatteRenderer,
    run_pet_overlay_qt,
)
from withnuri.pipeline.display_stabilizer import PetDisplayStabilizer
from withnuri.pipeline.inference_gate import IdleInferenceGate
from withnuri.pipeline.pet_crop import PetCropController
from withnuri.ui.debug_window import QtDebugWindow, run_tracking_debug_preview
from withnuri.ui.windowing import PreviewWindowUnavailable


_OVERLAY_QUALITY_PROFILES = {
    "balanced": {"image_size": 768, "decode_fps": 8, "mask_morphology_radius": 1},
    "low-power": {"image_size": 640, "decode_fps": 6, "mask_morphology_radius": 0},
    "high": {"image_size": 960, "decode_fps": 8, "mask_morphology_radius": 1},
}


def main(
    argv: Sequence[str] | None = None,
    *,
    runner=subprocess.run,
    decoder_factory=FfmpegFrameDecoder,
    yolo_tracker_factory=YoloPetTracker,
    debug_window_factory=QtDebugWindow,
    debug_runner=run_tracking_debug_preview,
    overlay_window_factory=OverlayWindow,
    overlay_runner=run_pet_overlay_qt,
    source_selector=choose_overlay_source,
    local_demo_session_factory=LocalDemoSession,
    local_relay_session_factory=LocalRelaySession,
    clock=time.monotonic,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        selection = source_selector(args.url, args.quality)
        if selection is None:
            return 0
        if isinstance(selection, OverlayLaunchSelection):
            args.url = selection.url
            args.quality = selection.quality
            args.start_local_demo = selection.start_local_demo
            args.start_local_relay = selection.start_local_relay
        else:  # Keeps injectable legacy selectors simple for tests/tools.
            args.url = selection
            args.quality = args.quality or "balanced"
            args.start_local_demo = False
            args.start_local_relay = False

    if args.command in {"overlay", "demo"}:
        _apply_overlay_quality_profile(args)

    if args.command == "moblin-rtmp":
        return _run_moblin_rtmp(args)
    if args.command == "mp4-rtmp":
        return _run_mp4_rtmp(args, runner=runner)
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
    if args.command in {"overlay", "demo"}:
        if args.command == "demo" and args.start_local_demo:
            return _run_local_demo_overlay(
                args,
                decoder_factory=decoder_factory,
                yolo_tracker_factory=yolo_tracker_factory,
                overlay_window_factory=overlay_window_factory,
                overlay_runner=overlay_runner,
                session_factory=local_demo_session_factory,
            )
        if args.command == "demo" and args.start_local_relay:
            return _run_local_relay_overlay(
                args,
                decoder_factory=decoder_factory,
                yolo_tracker_factory=yolo_tracker_factory,
                overlay_window_factory=overlay_window_factory,
                overlay_runner=overlay_runner,
                session_factory=local_relay_session_factory,
            )
        return _run_overlay(
            args,
            decoder_factory=decoder_factory,
            yolo_tracker_factory=yolo_tracker_factory,
            overlay_window_factory=overlay_window_factory,
            overlay_runner=overlay_runner,
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

    mp4_parser = subparsers.add_parser(
        "mp4-rtmp",
        help="Publish an MP4 file to MediaMTX over RTMP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mp4_parser.add_argument(
        "input",
        nargs="?",
        default="tests/video/dogcam.mp4",
        help="MP4 file to publish.",
    )
    mp4_parser.add_argument("--host", default="127.0.0.1", help="MediaMTX host.")
    mp4_parser.add_argument("--stream-name", default="nuri", help="MediaMTX path name.")
    mp4_parser.add_argument("--rtmp-port", type=int, default=1935, help="RTMP port.")
    mp4_parser.add_argument(
        "--rtsp-port", type=int, default=8554, help="RTSP readback port."
    )
    mp4_parser.add_argument("--width", type=int, default=1280, help="Published width.")
    mp4_parser.add_argument("--height", type=int, default=720, help="Published height.")
    mp4_parser.add_argument("--fps", type=float, default=30, help="Published frame rate.")
    mp4_parser.add_argument("--ffmpeg-binary", default="ffmpeg", help="ffmpeg executable.")
    mp4_loop_group = mp4_parser.add_mutually_exclusive_group()
    mp4_loop_group.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Repeat the MP4 indefinitely (default).",
    )
    mp4_loop_group.add_argument(
        "--once",
        dest="loop",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Publish the MP4 once, then exit.",
    )

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
        help="Show a camera debug window with YOLO masks and tracker IDs.",
    )
    tracking_parser.add_argument("url")
    _add_frame_size_args(tracking_parser)
    tracking_parser.add_argument("--display-width", type=int)
    tracking_parser.add_argument("--display-height", type=int)
    tracking_parser.add_argument("--decode-fps", type=float, default=8)
    tracking_parser.add_argument("--max-frames", type=int)
    tracking_parser.add_argument("--confidence", type=float, default=0.15)
    tracking_parser.add_argument("--image-size", type=int, default=768)
    tracking_parser.add_argument("--tracker-config", default="bytetrack.yaml")
    _add_device_arg(tracking_parser)
    tracking_parser.add_argument("--mask-cache-frames", type=int, default=8)
    tracking_parser.add_argument(
        "--diagnostics", action="store_true", help="Print pipeline timing and tracking stats."
    )

    overlay_parser = subparsers.add_parser(
        "overlay",
        help="Show the transparent always-on-top pet overlay.",
    )
    overlay_parser.add_argument("url")
    _add_overlay_options(overlay_parser, quality_default="balanced")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Choose a saved RTSP source, then show the pet overlay.",
    )
    demo_parser.add_argument(
        "--url",
        help="Pre-fill the launch-time source picker with an RTSP address.",
    )
    _add_overlay_options(demo_parser, quality_default=None)
    # The generic CLI helpers use a tiny frame for probes. A demo should match
    # the established 1280x720 overlay command instead of silently upscaling
    # a 320x180 decode.
    demo_parser.set_defaults(width=1280, height=720)

    return parser


def _add_overlay_options(
    overlay_parser: argparse.ArgumentParser, *, quality_default: str | None
) -> None:
    _add_frame_size_args(overlay_parser)
    overlay_parser.add_argument("--panel-width", type=int, default=480)
    overlay_parser.add_argument("--panel-height", type=int, default=270)
    overlay_parser.add_argument("--feather-radius", type=float, default=1.5)
    overlay_parser.add_argument(
        "--quality", choices=tuple(_OVERLAY_QUALITY_PROFILES), default=quality_default
    )
    overlay_parser.add_argument("--decode-fps", type=float)
    overlay_parser.add_argument("--max-frames", type=int)
    overlay_parser.add_argument("--confidence", type=float, default=0.15)
    overlay_parser.add_argument("--image-size", type=int)
    overlay_parser.add_argument("--tracker-config", default="bytetrack.yaml")
    _add_device_arg(overlay_parser)
    overlay_parser.add_argument("--mask-cache-frames", type=int, default=8)
    overlay_parser.add_argument("--lost-hold-seconds", type=float, default=0.6)
    overlay_parser.add_argument("--lost-fade-seconds", type=float, default=0.5)
    overlay_parser.add_argument("--display-confirm-frames", type=int, default=2)
    overlay_parser.add_argument("--display-grace-frames", type=int, default=8)
    overlay_parser.add_argument("--display-reassociation-iou", type=float, default=0.2)
    overlay_parser.add_argument(
        "--display-reassociation-distance", type=float, default=1.25
    )
    overlay_parser.add_argument("--crop-padding", type=float, default=0.35)
    overlay_parser.add_argument("--crop-smoothing", type=float, default=0.35)
    overlay_parser.add_argument("--mask-temporal-weight", type=float, default=0.25)
    overlay_parser.add_argument("--mask-morphology-radius", type=int)
    overlay_parser.add_argument(
        "--idle-inference-fps",
        type=float,
        default=2.0,
        help="Inference FPS while no pet is active; use 0 to disable throttling.",
    )
    overlay_parser.add_argument(
        "--reconnect-delay-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait before retrying a disconnected RTSP stream.",
    )
    overlay_parser.add_argument(
        "--diagnostics", action="store_true", help="Print pipeline timing and tracking stats."
    )

def _add_frame_size_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)


def _add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="YOLO inference device. auto prefers CUDA, then CPU; MPS is opt-in.",
    )


def _apply_overlay_quality_profile(args) -> None:
    profile = _OVERLAY_QUALITY_PROFILES[args.quality]
    for option, value in profile.items():
        if getattr(args, option) is None:
            setattr(args, option, value)


def _report_missing_decode_tool() -> int:
    print(
        "ffmpeg was not found. Install ffmpeg or configure a bundled decode tool.",
        file=sys.stderr,
    )
    return 2


def _run_moblin_rtmp(args) -> int:
    profile = MoblinRtmpProfile(host=args.host, stream_name=args.stream_name)
    print(f"Publish URL: {profile.publish_url}")
    print(f"WithNuri read URL: {profile.read_url}")
    print()
    print("Moblin setup:")
    for index, instruction in enumerate(profile.instructions, start=1):
        print(f"{index}. {instruction}")
    return 0


def _run_mp4_rtmp(args, *, runner) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"MP4 file not found: {args.input}", file=sys.stderr)
        return 1

    command = _build_mp4_rtmp_command(args)
    print("Publishing MP4 to MediaMTX over RTMP.")
    print(f"Input: {args.input}")
    print(f"Publish URL: {_mp4_rtmp_publish_url(args)}")
    print(f"WithNuri read URL: {_mp4_rtmp_read_url(args)}")
    print("Press Ctrl-C to stop.")

    try:
        completed = runner(command)
    except FileNotFoundError:
        return _report_missing_decode_tool()
    except KeyboardInterrupt:
        print("MP4 RTMP publish interrupted.")
        return 130

    if completed.returncode == 0:
        return 0
    if completed.returncode in {130, 255}:
        print("MP4 RTMP publish interrupted.")
        return 130

    print(
        f"MP4 RTMP publish failed with exit code {completed.returncode}.",
        file=sys.stderr,
    )
    return completed.returncode


def _build_mp4_rtmp_command(args) -> list[str]:
    video_filter = (
        f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
        f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={args.fps:g},format=yuv420p"
    )
    command = [args.ffmpeg_binary, "-re"]
    if getattr(args, "loop", True):
        command.extend(["-stream_loop", "-1"])
    command.extend(
        [
            "-i",
            args.input,
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-f",
            "flv",
            _mp4_rtmp_publish_url(args),
        ]
    )
    return command


def _mp4_rtmp_publish_url(args) -> str:
    return f"rtmp://{args.host}:{args.rtmp_port}/{args.stream_name}"


def _mp4_rtmp_read_url(args) -> str:
    return f"rtsp://{args.host}:{args.rtsp_port}/{args.stream_name}"


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
        return _report_missing_decode_tool()
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
        return _report_missing_decode_tool()
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
            tracker_config=args.tracker_config,
            cache_frames=args.mask_cache_frames,
            device=args.device,
        )
        print(f"Inference device: {tracker.inference_device}")
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
        runner_kwargs = {"max_frames": args.max_frames}
        if args.diagnostics:
            runner_kwargs["collect_diagnostics"] = True
        result = debug_runner(decoder, tracker=tracker, window=window, **runner_kwargs)
    except DecodeToolMissing:
        return _report_missing_decode_tool()
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
        _print_tracking_debug_result(
            "interrupted", args.url, result, tracker_name=tracker.display_name
        )
        if args.diagnostics:
            _print_pipeline_diagnostics(result)
        return 130
    _print_tracking_debug_result(
        "closed", args.url, result, tracker_name=tracker.display_name
    )
    if args.diagnostics:
        _print_pipeline_diagnostics(result)
    return 0


def _run_overlay(
    args,
    *,
    decoder_factory,
    yolo_tracker_factory,
    overlay_window_factory,
    overlay_runner,
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
            tracker_config=args.tracker_config,
            cache_frames=args.mask_cache_frames,
            device=args.device,
        )
        print(f"Inference device: {tracker.inference_device}")
        window = overlay_window_factory(
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
        crop_controller = PetCropController(
            output_width=args.panel_width,
            output_height=args.panel_height,
            padding_ratio=args.crop_padding,
            smoothing=args.crop_smoothing,
        )
        set_panel_resize_handler = getattr(window, "set_panel_resize_handler", None)
        if callable(set_panel_resize_handler):
            set_panel_resize_handler(crop_controller.set_output_size)
        runner_kwargs = {
            "tracker": tracker,
            "window": window,
            "fade": FadeController(
                hold_seconds=args.lost_hold_seconds,
                fade_seconds=args.lost_fade_seconds,
            ),
            "stabilizer": PetDisplayStabilizer(
                confirm_frames=args.display_confirm_frames,
                grace_frames=args.display_grace_frames,
                reassociation_iou=args.display_reassociation_iou,
                reassociation_center_distance=args.display_reassociation_distance,
            ),
            "crop_controller": crop_controller,
            "renderer": TemporalMatteRenderer(
                temporal_weight=args.mask_temporal_weight,
                morphology_radius=args.mask_morphology_radius,
            ),
            "inference_gate": IdleInferenceGate(
                idle_fps=args.idle_inference_fps
            ),
            "quality_profile": args.quality,
            "source_label": _overlay_source_label(args.url),
            "feather_radius": args.feather_radius,
            "max_frames": args.max_frames,
            "reconnect_delay_seconds": args.reconnect_delay_seconds,
            "on_stream_status": _overlay_stream_status_printer(
                reconnect_delay_seconds=args.reconnect_delay_seconds
            ),
        }
        if args.diagnostics:
            runner_kwargs["collect_diagnostics"] = True
        result = overlay_runner(decoder, **runner_kwargs)
    except DecodeToolMissing:
        return _report_missing_decode_tool()
    except YoloTrackingDependencyMissing as exc:
        print(f"YOLO tracking is unavailable: {exc}", file=sys.stderr)
        return 2
    except PreviewWindowUnavailable as exc:
        print(f"Overlay window is unavailable: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Overlay failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Overlay interrupted.")
        return 130

    status = "interrupted" if result.interrupted else "closed"
    print(f"Overlay {status}: {args.url}")
    print(f"Tracker: {tracker.display_name}")
    print(f"Frames rendered: {result.frames_rendered}")
    if args.diagnostics:
        _print_pipeline_diagnostics(result)
    return 130 if result.interrupted else 0


def _run_local_demo_overlay(
    args,
    *,
    decoder_factory,
    yolo_tracker_factory,
    overlay_window_factory,
    overlay_runner,
    session_factory,
) -> int:
    session = session_factory()
    try:
        print("Starting local demo stream.")
        session.start()
        return _run_overlay(
            args,
            decoder_factory=decoder_factory,
            yolo_tracker_factory=yolo_tracker_factory,
            overlay_window_factory=overlay_window_factory,
            overlay_runner=overlay_runner,
        )
    except LocalDemoUnavailable as exc:
        print(f"Local demo unavailable: {exc}", file=sys.stderr)
        return 2
    finally:
        session.stop()


def _run_local_relay_overlay(
    args,
    *,
    decoder_factory,
    yolo_tracker_factory,
    overlay_window_factory,
    overlay_runner,
    session_factory,
) -> int:
    """Run the overlay against a relay owned by this app, without a publisher."""
    session = session_factory()
    try:
        print("Starting local Moblin relay. Waiting for the phone stream.")
        session.start()
        return _run_overlay(
            args,
            decoder_factory=decoder_factory,
            yolo_tracker_factory=yolo_tracker_factory,
            overlay_window_factory=overlay_window_factory,
            overlay_runner=overlay_runner,
        )
    except LocalDemoUnavailable as exc:
        print(f"Local relay unavailable: {exc}", file=sys.stderr)
        return 2
    finally:
        session.stop()


def _overlay_stream_status_printer(*, reconnect_delay_seconds: float):
    def report(status: str, detail: str | None = None) -> None:
        if status == "reconnecting":
            suffix = f": {detail}" if detail else ""
            print(
                "Stream reconnecting "
                f"in {reconnect_delay_seconds:g}s{suffix}",
                flush=True,
            )
        elif status == "live":
            print("Stream reconnected.", flush=True)

    return report


def _overlay_source_label(url: str) -> str:
    """Keep the tray useful without exposing stream credentials."""
    parsed = urlsplit(url)
    host = parsed.hostname or parsed.netloc
    path = parsed.path.lstrip("/")
    if host in {"127.0.0.1", "localhost", "::1"} and path == "nuri":
        return "Local relay"
    if host and path:
        return f"{host}/{path}"
    return host or "Current stream"


def _print_tracking_debug_result(
    status: str, url: str, result, *, tracker_name: str
) -> None:
    print(f"Tracking debug {status}: {url}")
    print(f"Tracker: {tracker_name}")
    print(f"Frames rendered: {result.frames_rendered}")
    if status == "interrupted":
        print(
            f"Visible pet frames: {result.visible_frame_count}/{result.frames_rendered}"
        )
        print(f"Last frame visible: {'yes' if result.last_frame_visible else 'no'}")


def _print_pipeline_diagnostics(result) -> None:
    diagnostics = result.diagnostics
    if diagnostics is None:
        print("Diagnostics: unavailable")
        return
    print("Diagnostics:")
    print(
        "  Frames: "
        f"decoded {diagnostics.decoded_frame_count} ({diagnostics.decoded_fps:.1f} fps), "
        f"processed {diagnostics.processed_frame_count} ({diagnostics.processed_fps:.1f} fps), "
        f"dropped {diagnostics.dropped_frame_count}, "
        f"idle skipped {diagnostics.skipped_inference_frame_count}"
    )
    print(
        "  Timing: "
        f"queue {diagnostics.average_queue_delay_ms:.1f}/{diagnostics.max_queue_delay_ms:.1f} ms, "
        f"inference {diagnostics.average_inference_ms:.1f}/{diagnostics.max_inference_ms:.1f} ms, "
        f"render {diagnostics.average_render_ms:.1f}/{diagnostics.max_render_ms:.1f} ms, "
        f"present {diagnostics.average_present_ms:.1f}/{diagnostics.max_present_ms:.1f} ms "
        "(avg/max)"
    )
    print(
        "  Tracking: "
        f"detected {diagnostics.detected_frame_count}/{diagnostics.processed_frame_count}, "
        f"lost {diagnostics.track_loss_events}, "
        f"recovered {diagnostics.track_recovery_events}, "
        f"ID changes {diagnostics.track_id_change_events}"
    )
    display = diagnostics.display_metrics
    if display is not None:
        print(
            "  Display stability: "
            f"suppressed {display.suppressed_detection_frames}, "
            f"confirmed {display.confirmed_tracks}, "
            f"re-associated {display.reassociation_events}, "
            f"grace retained {display.grace_retained_frames}, "
            f"expired {display.expired_tracks}"
        )
