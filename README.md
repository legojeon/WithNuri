# WithNuri

A desktop prototype that reads a live pet camera stream, runs YOLO11n-seg with ByteTrack, and draws dog tracking diagnostics in a debug window.

Flow: phone camera (Moblin, RTMP) → MediaMTX (RTMP in, RTSP out) → WithNuri (reads RTSP). The phone and computer must be on the same Wi-Fi/LAN.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,ml,ui]"
```

Requires `ffmpeg`/`ffprobe` on your `PATH` (e.g. `brew install ffmpeg`).

For copy-paste command examples, see [`COMMANDS.md`](COMMANDS.md).

## MediaMTX

The config (`mediamtx.yml`) is in `WithNuri-tools/mediamtx/`, but the binary is not committed. Download it for your OS from https://github.com/bluenviron/mediamtx/releases, place the `mediamtx` binary next to `mediamtx.yml`, then run:

```bash
cd WithNuri-tools/mediamtx
./mediamtx mediamtx.yml
```

## Stream from the phone

Find your computer's LAN IP (`ipconfig getifaddr en0`), then generate Moblin RTMP values:

```bash
.venv/bin/python -m withnuri moblin-rtmp --host <LAN-IP> --stream-name nuri
```

Enter the printed RTMP URL in Moblin (RTMP protocol) and start streaming.

## Stream from an MP4

To test the same MediaMTX RTMP/RTSP path without Moblin, start MediaMTX and run:

```bash
.venv/bin/python -m withnuri mp4-rtmp --loop
```

This loops `tests/video/dogcam.mp4` into `rtmp://127.0.0.1:1935/nuri`.
WithNuri reads the relayed stream at `rtsp://127.0.0.1:8554/nuri`.

## Tracking debug window

With MediaMTX running and the phone streaming:

```bash
.venv/bin/python -m withnuri tracking-debug rtsp://<LAN-IP>:8554/nuri \
  --width 1280 --height 720 \
  --display-width 1280 --display-height 720 \
  --image-size 960 \
  --decode-fps 8 \
  --confidence 0.2 \
  --mask-cache-frames 8
```

Higher `--width`/`--height`/`--image-size` detect distant dogs better but lower FPS.
