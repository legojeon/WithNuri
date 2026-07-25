$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Ffmpeg = if ($env:FFMPEG_BIN) { $env:FFMPEG_BIN } else { "ffmpeg.exe" }
$MediaMtx = Join-Path $Root "WithNuri-tools\mediamtx\mediamtx.exe"
$MediaMtxConfig = Join-Path $Root "WithNuri-tools\mediamtx\mediamtx.yml"
$DemoVideo = Join-Path $Root "tests\video\dogcam.mp4"
$YoloModel = Join-Path $Root "yolo11n-seg.pt"
$AppIcon = Join-Path $Root "assets\WithNuri.ico"
$TrayIcon = Join-Path $Root "assets\withnuri-tray.svg"

foreach ($Asset in @($Python, $MediaMtx, $MediaMtxConfig, $DemoVideo, $YoloModel, $AppIcon, $TrayIcon)) {
    if (-not (Test-Path $Asset)) {
        throw "Required packaging asset is missing: $Asset"
    }
}

Set-Location $Root
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name WithNuri `
    --icon $AppIcon `
    --paths src `
    --collect-all ultralytics `
    --exclude-module polars `
    --add-data "$YoloModel;." `
    --add-data "$MediaMtxConfig;assets\mediamtx" `
    --add-data "$DemoVideo;assets" `
    --add-data "$TrayIcon;assets" `
    --add-binary "$MediaMtx;assets\mediamtx" `
    --add-binary "$Ffmpeg;bin" `
    src\withnuri\app\demo.py

Write-Output "Built: $Root\dist\WithNuri\WithNuri.exe"
