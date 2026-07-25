#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg || true)}"
MEDIAMTX_BIN="$ROOT/WithNuri-tools/mediamtx/mediamtx"
MEDIAMTX_CONFIG="$ROOT/WithNuri-tools/mediamtx/mediamtx.yml"
DEMO_VIDEO="$ROOT/tests/video/dogcam.mp4"
YOLO_MODEL="$ROOT/yolo11n-seg.pt"
APP_ICON="$ROOT/assets/WithNuri.icns"
TRAY_ICON="$ROOT/assets/withnuri-tray.svg"

# Keep PyInstaller and Matplotlib caches inside the project. This makes the
# script work in sandboxed/CI environments where the user Library is read-only.
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$ROOT/.cache/pyinstaller}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"
mkdir -p "$PYINSTALLER_CONFIG_DIR" "$MPLCONFIGDIR"

for asset in "$FFMPEG_BIN" "$MEDIAMTX_BIN" "$MEDIAMTX_CONFIG" "$DEMO_VIDEO" "$YOLO_MODEL" "$APP_ICON" "$TRAY_ICON"; do
  if [[ -z "$asset" || ! -f "$asset" ]]; then
    echo "Required packaging asset is missing: $asset" >&2
    exit 2
  fi
done

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This build script currently bundles Apple Silicon binaries only." >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name WithNuri \
  --icon "$APP_ICON" \
  --paths src \
  --collect-all ultralytics \
  --exclude-module polars \
  --add-data "$YOLO_MODEL:." \
  --add-data "$MEDIAMTX_CONFIG:assets/mediamtx" \
  --add-data "$DEMO_VIDEO:assets" \
  --add-data "$TRAY_ICON:assets" \
  --add-binary "$MEDIAMTX_BIN:assets/mediamtx" \
  --add-binary "$FFMPEG_BIN:bin" \
  src/withnuri/app/demo.py

echo "Built: $ROOT/dist/WithNuri.app"
