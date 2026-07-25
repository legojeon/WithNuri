"""Double-click entry point for the packaged WithNuri demo app."""

from withnuri.app.main import main as cli_main


def main() -> int:
    return cli_main(["demo"])


if __name__ == "__main__":
    raise SystemExit(main())
