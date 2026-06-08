"""Command-line entry point (stub).

``hivevision <image>`` will eventually read a position from a photo. The ML
pipeline is not built yet, so for now this just exposes the version and points at
the capture/label app, which is the working tool today.
"""

from __future__ import annotations

import argparse
import sys

from hivevision import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hivevision", description=__doc__)
    parser.add_argument("--version", action="version", version=f"hivevision {__version__}")
    parser.add_argument(
        "image", nargs="?", help="photo to read a position from (not implemented yet)"
    )
    args = parser.parse_args(argv)

    if args.image:
        print("Reading positions from photos isn't implemented yet (see plan.md phases 3-4).")
        print("To label photos now:  uv run python -m hivevision.capture")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
