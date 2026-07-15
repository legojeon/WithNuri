# WithNuri Commands

Run these commands from the project root unless a section says otherwise.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,ml,ui]"
```

Check ffmpeg:

```bash
ffmpeg -version
ffprobe -version
```

## MediaMTX

Start MediaMTX:

```bash
cd WithNuri-tools/mediamtx
./mediamtx mediamtx.yml
```

If the binary is not executable:

```bash
chmod +x WithNuri-tools/mediamtx/mediamtx
```

## Phone / Moblin RTMP

Find the Mac LAN IP:

```bash
ipconfig getifaddr en0
```

Print Moblin RTMP values:

```bash
.venv/bin/python -m withnuri moblin-rtmp --host <LAN-IP> --stream-name nuri
```

Use the printed `Publish URL` in Moblin. WithNuri reads:

```text
rtsp://<LAN-IP>:8554/nuri
```

## MP4 to RTMP

Use this to test the full MediaMTX RTMP/RTSP path without Moblin. Start MediaMTX
first, then run:

```bash
.venv/bin/python -m withnuri mp4-rtmp --loop
```

`--loop` repeats the MP4 indefinitely and is the default. Use `--once` when a
single playback is needed.

To use another MP4:

```bash
.venv/bin/python -m withnuri mp4-rtmp /path/to/dog-video.mp4
```

WithNuri reads the relayed stream at:

```text
rtsp://127.0.0.1:8554/nuri
```

### Reconnect test

Keep the overlay running, stop the `mp4-rtmp --loop` process with `Ctrl-C`,
then start it again. The overlay window stays open, prints `Stream reconnecting`
while it retries every 2 seconds, and prints `Stream reconnected.` when frames
return. Set a shorter retry interval only for this test if needed:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --reconnect-delay-seconds 1 --diagnostics
```

### Apple Silicon MPS comparison

On an Apple Silicon Mac where `torch.backends.mps.is_available()` is `True`,
compare GPU inference against CPU with the same quality profile:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --image-size 640 --decode-fps 6 --mask-morphology-radius 0 \
  --device mps --diagnostics
```

`--device auto` (the default) prefers CUDA, then CPU. MPS is opt-in because
the current Apple Silicon benchmark was slower and showed no clear thermal
benefit; use `--device cpu` for a direct baseline comparison.

### Quality profiles and idle power saving

`--quality balanced` (default), `--quality low-power`, and `--quality high`
select tested starting values. Hardware differs, so any explicit advanced
option overrides its profile; for example, use `--quality low-power
--decode-fps 7` on a faster machine.

When no pet is active, overlay defaults to 2 inference FPS to save CPU. It
returns to the selected decode FPS as soon as a pet is detected. Use
`--idle-inference-fps 0` to disable this adaptive behavior for benchmarking.

## Debug Window

Debug window from local MediaMTX:

```bash
.venv/bin/python -m withnuri tracking-debug rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --display-width 1280 --display-height 720 \
  --image-size 960 \
  --decode-fps 8 \
  --confidence 0.2 \
  --mask-cache-frames 8
```

Debug window from phone/Moblin:

```bash
.venv/bin/python -m withnuri tracking-debug rtsp://<LAN-IP>:8554/nuri \
  --width 1280 --height 720 \
  --display-width 1280 --display-height 720 \
  --image-size 960 \
  --decode-fps 8 \
  --confidence 0.2 \
  --mask-cache-frames 8
```

## Transparent Overlay

Detailed benchmark results and the rationale for the default profile are in
[`docs/overlay-quality-benchmark.md`](docs/overlay-quality-benchmark.md).
The tray-control UX direction is documented in
[`docs/overlay-control-ux.md`](docs/overlay-control-ux.md).

### Recommended balanced preset

The following is the current balanced preset for the MacBook Air test setup.
It uses a 768px inference image to reduce dropped frames while retaining stable
segmentation quality.

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --panel-width 480 --panel-height 270 \
  --image-size 768 \
  --decode-fps 8 \
  --confidence 0.15 \
  --lost-hold-seconds 0.6 \
  --lost-fade-seconds 0.5
```

### Quality presets

Use one of these presets according to the camera framing and available compute.
The overlay keeps a padded, smoothed crop around the detected pets by default.
It also applies a light per-track mask cleanup and temporal blend
(`--mask-morphology-radius 1`, `--mask-temporal-weight 0.25`). Set both values
to `0` when comparing an unfiltered mask during debugging.

| Preset | Use when | Key options |
| --- | --- | --- |
| Fast | Battery/CPU headroom is limited. | `--image-size 640 --decode-fps 10` |
| Balanced (default) | Normal desk use with a clearly visible pet. | `--image-size 768 --decode-fps 8` |
| High quality | The pet is small or far from the camera. | `--image-size 960 --decode-fps 8` |

Fast preset:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --image-size 640 --decode-fps 10
```

High-quality preset:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --image-size 960 --decode-fps 8
```

Transparent overlay from local MediaMTX:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --panel-width 480 --panel-height 270 \
  --image-size 768 \
  --decode-fps 8 \
  --confidence 0.15 \
  --lost-hold-seconds 0.6 \
  --lost-fade-seconds 0.5 \
  --feather-radius 1.5
```

### Diagnostics benchmark

Use the looping `tests/video/dogcam.mp4` relay from the **MP4 to RTMP** section
as a repeatable baseline. Measure 300 frames and compare `dropped`, `lost`, and
`ID changes` before changing a preset or tracker.

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --image-size 768 \
  --decode-fps 8 \
  --max-frames 300 \
  --diagnostics
```

ByteTrack is the current default. In the MacBook Air 300-frame test, BoT-SORT
kept the same detection rate but increased ID changes from 14 to 19. Re-run the
comparison after changing the camera, model, or tracker configuration:

```bash
.venv/bin/python -m withnuri overlay rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720 \
  --image-size 768 \
  --decode-fps 8 \
  --tracker-config botsort.yaml \
  --max-frames 300 \
  --diagnostics
```

Transparent overlay from phone/Moblin:

```bash
.venv/bin/python -m withnuri overlay rtsp://<LAN-IP>:8554/nuri \
  --width 1280 --height 720 \
  --panel-width 480 --panel-height 270 \
  --image-size 768 \
  --decode-fps 8 \
  --confidence 0.15 \
  --lost-hold-seconds 0.6 \
  --lost-fade-seconds 0.5 \
  --feather-radius 1.5
```

## Stream Checks

Check whether RTSP is reachable:

```bash
.venv/bin/python -m withnuri probe rtsp://127.0.0.1:8554/nuri
```

Read frames for a few seconds:

```bash
.venv/bin/python -m withnuri stream-check rtsp://127.0.0.1:8554/nuri \
  --seconds 3 \
  --width 1280 --height 720
```

Grab one frame:

```bash
.venv/bin/python -m withnuri grab-frame rtsp://127.0.0.1:8554/nuri \
  --width 1280 --height 720
```

## Suggested Defaults

For normal RTMP-relayed dog-video testing:

```text
video: 1280x720 MP4, 30 fps, H.264 if available
--width 1280 --height 720
--image-size 960
--decode-fps 8
--confidence 0.2
```

Use `Ctrl-C` to stop long-running commands.
