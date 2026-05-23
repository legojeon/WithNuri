from withnuri.app.main import main


def test_main_prints_moblin_rtmp_setup(capsys):
    exit_code = main(["moblin-rtmp", "--host", "relay.example.com", "--stream-name", "nuri"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Publish URL: rtmp://relay.example.com:1935/nuri\n"
        "WithNuri read URL: rtsp://relay.example.com:8554/nuri\n"
        "\n"
        "Moblin setup:\n"
        "1. Open Moblin stream settings.\n"
        "2. Choose RTMP as the streaming protocol.\n"
        "3. Set the RTMP URL to rtmp://relay.example.com:1935/nuri.\n"
        "4. Start streaming, then connect WithNuri to rtsp://relay.example.com:8554/nuri.\n"
    )
