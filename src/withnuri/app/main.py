import argparse
from collections.abc import Sequence

from withnuri.streaming.phone_profiles import MoblinRtmpProfile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="withnuri")
    subparsers = parser.add_subparsers(dest="command", required=True)

    moblin_parser = subparsers.add_parser("moblin-rtmp", help="Print Moblin RTMP setup values.")
    moblin_parser.add_argument("--host", required=True)
    moblin_parser.add_argument("--stream-name", required=True)

    args = parser.parse_args(argv)
    if args.command == "moblin-rtmp":
        profile = MoblinRtmpProfile(host=args.host, stream_name=args.stream_name)
        print(f"Publish URL: {profile.publish_url}")
        print(f"WithNuri read URL: {profile.read_url}")
        print()
        print("Moblin setup:")
        for index, instruction in enumerate(profile.instructions, start=1):
            print(f"{index}. {instruction}")
        return 0

    parser.error(f"unknown command: {args.command}")
