class IdleInferenceGate:
    """Runs every frame while a pet is active and throttles empty scenes."""

    def __init__(self, *, idle_fps: float = 2.0) -> None:
        if idle_fps < 0:
            raise ValueError("idle_fps must not be negative")
        self._idle_fps = idle_fps
        self._last_inference_timestamp: float | None = None
        self._pet_is_active = False

    def should_infer(self, timestamp_seconds: float) -> bool:
        if self._idle_fps == 0 or self._pet_is_active:
            return True
        if self._last_inference_timestamp is None:
            return True
        return timestamp_seconds - self._last_inference_timestamp >= 1 / self._idle_fps

    def record_inference(self, *, timestamp_seconds: float, pet_is_active: bool) -> None:
        self._last_inference_timestamp = timestamp_seconds
        self._pet_is_active = pet_is_active

    def reset(self) -> None:
        self._last_inference_timestamp = None
        self._pet_is_active = False
