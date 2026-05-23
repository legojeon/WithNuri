from dataclasses import dataclass
import re


_STREAM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class PhoneRtmpProfile:
    host: str
    stream_name: str
    rtmp_port: int = 1935
    rtsp_port: int = 8554

    def __post_init__(self) -> None:
        if not self.host or "/" in self.host:
            raise ValueError("host must be a hostname or IP address without a path")
        if not _STREAM_NAME_PATTERN.fullmatch(self.stream_name):
            raise ValueError("stream_name must contain only letters, numbers, underscores, or hyphens")

    @property
    def publish_url(self) -> str:
        return f"rtmp://{self.host}:{self.rtmp_port}/{self.stream_name}"

    @property
    def read_url(self) -> str:
        return f"rtsp://{self.host}:{self.rtsp_port}/{self.stream_name}"


class MoblinRtmpProfile(PhoneRtmpProfile):

    @property
    def instructions(self) -> list[str]:
        return [
            "Open Moblin stream settings.",
            "Choose RTMP as the streaming protocol.",
            f"Set the RTMP URL to {self.publish_url}.",
            f"Start streaming, then connect WithNuri to {self.read_url}.",
        ]


class YaseaRtmpProfile(PhoneRtmpProfile):

    @property
    def instructions(self) -> list[str]:
        return [
            "Open the Android RTMP publisher settings.",
            f"Set the RTMP URL to {self.publish_url}.",
            f"Start streaming, then connect WithNuri to {self.read_url}.",
        ]
