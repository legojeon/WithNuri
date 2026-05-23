import pytest

from withnuri.streaming.phone_profiles import MoblinRtmpProfile, YaseaRtmpProfile


def test_moblin_rtmp_profile_builds_publish_and_read_urls_for_mediamtx():
    profile = MoblinRtmpProfile(host="relay.example.com", stream_name="nuri")

    assert profile.publish_url == "rtmp://relay.example.com:1935/nuri"
    assert profile.read_url == "rtsp://relay.example.com:8554/nuri"
    assert profile.instructions == [
        "Open Moblin stream settings.",
        "Choose RTMP as the streaming protocol.",
        "Set the RTMP URL to rtmp://relay.example.com:1935/nuri.",
        "Start streaming, then connect WithNuri to rtsp://relay.example.com:8554/nuri.",
    ]


@pytest.mark.parametrize(
    ("host", "stream_name"),
    [
        ("", "nuri"),
        ("relay.example.com", ""),
        ("relay.example.com/live", "nuri"),
        ("relay.example.com", "nuri/cam"),
        ("relay.example.com", "nuri cam"),
    ],
)
def test_moblin_rtmp_profile_rejects_ambiguous_url_parts(host, stream_name):
    with pytest.raises(ValueError):
        MoblinRtmpProfile(host=host, stream_name=stream_name)


def test_yasea_rtmp_profile_uses_same_relay_urls_with_android_instructions():
    profile = YaseaRtmpProfile(host="relay.example.com", stream_name="nuri")

    assert profile.publish_url == "rtmp://relay.example.com:1935/nuri"
    assert profile.read_url == "rtsp://relay.example.com:8554/nuri"
    assert profile.instructions == [
        "Open the Android RTMP publisher settings.",
        "Set the RTMP URL to rtmp://relay.example.com:1935/nuri.",
        "Start streaming, then connect WithNuri to rtsp://relay.example.com:8554/nuri.",
    ]
