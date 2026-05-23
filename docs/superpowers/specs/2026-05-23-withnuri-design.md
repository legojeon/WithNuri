# WithNuri Design

## Purpose

WithNuri is an open source desktop app that lets remote workers keep a lightweight visual presence of their dog on screen while working. A home camera or spare phone streams video from home, WithNuri removes the background locally on the desktop machine, and the dog appears as a transparent always-on-top overlay.

The first product goal is not surveillance or full home-camera replacement. The goal is emotional presence: the user should feel that their dog is nearby without keeping a separate camera app open.

## Reference Projects

- `backgroundremover`: https://github.com/nadermx/backgroundremover.git
- `MediaMTX`: https://github.com/bluenviron/mediamtx.git

`backgroundremover` is used as the first segmentation reference. Its file/video utilities are not used directly for live streaming; the useful parts are the U2Net model loading and batched mask inference structure.

`MediaMTX` is used as the recommended stream bridge. It can accept camera streams and expose them in protocols that are easier for a desktop app to consume remotely, especially HLS/LL-HLS and later WebRTC.

## MVP Scope

The MVP is a cross-platform desktop app for macOS and Windows.

Core features:

- Accept a remote stream URL.
- Prioritize HLS/LL-HLS URL input for the first working path.
- Decode frames at a reduced working resolution.
- Run local pet foreground segmentation.
- Smooth and lightly clean the alpha mask.
- Render the dog in a transparent, always-on-top desktop overlay.
- Allow the user to resize, move, show, hide, and reconnect the overlay.

Out of scope for the first MVP:

- Native mobile camera app.
- Account system.
- Cloud hosting.
- Premium or advanced feature tiers.
- Full camera vendor integrations.
- Guaranteed low-latency WebRTC support.
- Multi-pet identity tracking.

## Stream Architecture

The intended remote flow is:

```text
Home camera or spare phone
  -> MediaMTX at home
  -> Cloudflare Tunnel HTTPS hostname
  -> WithNuri desktop app at work
  -> transparent dog overlay
```

The first supported input should be an HTTPS HLS or LL-HLS URL exposed through MediaMTX and Cloudflare Tunnel. This keeps the app simple because the desktop client can read a normal URL without requiring users to run a separate client-side TCP tunnel command.

RTSP should remain useful for local development and advanced manual setups, but it should not be the primary user-facing remote path. WebRTC should be treated as a later input backend because it can reduce latency but is more complex to decode into frames for local ML processing.

## Desktop Architecture

The recommended implementation stack is Python with PySide6.

Components:

- `StreamReader`: opens the configured stream URL and yields decoded frames.
- `FrameSampler`: limits resolution and frame rate to keep CPU/GPU usage predictable.
- `SegmentationEngine`: stable interface for foreground mask inference.
- `U2NetSegmentationEngine`: first implementation, based on the useful model-loading and batched inference structure from `backgroundremover`.
- `MaskPostProcessor`: applies thresholding, light blur, temporal smoothing, and optional crop tracking.
- `OverlayWindow`: transparent always-on-top window that displays RGBA dog frames.
- `SettingsWindow`: minimal URL, reconnect, model, scale, opacity, and performance controls.

The segmentation engine must be replaceable. Dog segmentation can fail on fur, shadows, bedding, and similar-colored floors, so the code should allow later engines such as MediaPipe, YOLO segmentation, SAM-family models, or a pet-specific model without rewriting the desktop app.

## Data Flow

```text
Stream URL
  -> decode frame
  -> downscale frame
  -> infer mask
  -> smooth mask
  -> compose RGBA dog frame
  -> render overlay
```

The app should never call `backgroundremover.remove()` per frame in the live path. The model must be loaded once and reused. The first engine should prefer a lightweight model such as `u2netp` for MVP performance testing, while leaving room for higher-quality models.

## Performance Targets

Initial targets:

- 5-15 processed frames per second.
- 360p or 640px-wide internal inference resolution.
- 2-5 seconds of stream latency is acceptable for MVP if the connection is stable.
- UI must remain responsive even when inference is slow.

The app should degrade gracefully by reducing processing FPS, reducing inference resolution, or temporarily showing the latest valid frame when the stream or model stalls.

## Error Handling

The MVP should handle:

- Invalid stream URL.
- Stream timeout or disconnect.
- Media decode errors.
- Missing model files.
- Model download failure.
- Unsupported platform acceleration.
- Inference slower than input frame rate.

Errors should be visible in the settings window, while the overlay itself remains quiet and unobtrusive.

## Security And Privacy

WithNuri should process frames locally on the user's desktop. It should not upload video frames to any WithNuri-controlled service.

For remote access, the recommended open source setup is user-owned infrastructure:

- MediaMTX runs near the camera.
- Cloudflare Tunnel exposes the selected stream endpoint without opening inbound router ports.
- The desktop app consumes the resulting URL.

The project should document the tradeoffs clearly: URL exposure, Cloudflare account dependency, camera credentials, and the difference between HLS, RTSP, and WebRTC paths.

## Testing Strategy

The first test set should use recorded sample videos before live cameras:

- A dog moving across a static indoor background.
- A dog partially occluded by furniture or bedding.
- A dog with similar color to the floor or blanket.
- Empty room frames.
- Stream disconnect and reconnect simulation.

Automated tests should cover component boundaries where practical:

- URL/config validation.
- Frame sampler behavior.
- Segmentation engine interface contract with a fake engine.
- Mask post-processing on synthetic masks.
- Reconnect state transitions.

Manual verification is required for overlay behavior on macOS and Windows because transparency, always-on-top, click-through, and taskbar behavior are OS-specific.

## MVP Decisions

- The first overlay supports manual hide/show and dragging. Click-through is not required for the first MVP because OS-specific behavior can slow down the cross-platform build.
- The app downloads the model on first launch rather than bundling it in the initial repository. The setup flow must explain model size and local storage location.
- The first user setup guide documents HLS/LL-HLS through MediaMTX and Cloudflare Tunnel. RTSP can be documented for local development and manual testing.
- The first setup guide should use a spare-phone camera app only after a small compatibility check confirms a stable RTSP, RTMP, or HLS publishing path into MediaMTX.

## Follow-Up Research

- Pick one Android spare-phone camera app and one iOS spare-phone camera app for the first documented setup.
- Measure `u2netp` and `u2net` inference speed on Apple Silicon Mac, Intel Mac, and Windows CPU-only machines.
- Verify MediaMTX HLS/LL-HLS output behavior through Cloudflare Tunnel with a long-running stream.
