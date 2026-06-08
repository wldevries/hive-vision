"""Launch the capture/label web app.

    uv run python -m hivevision.capture

Drop phone photos in ``data/store/inbox/`` (subfolders OK), open the printed URL,
and mark each tile's icon center with its class. Labels are written to
``data/labels.jsonl`` and the photos EXIF-normalized into ``data/store/normalized/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from hivevision.capture.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m hivevision.capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 for LAN access)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="store root: photos from <root>/store/inbox/, labels to <root>/labels.jsonl",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    (args.root / "store" / "inbox").mkdir(parents=True, exist_ok=True)
    print(f"Open http://{args.host}:{args.port}  ·  inbox: {args.root / 'store' / 'inbox'}")
    uvicorn.run(create_app(args.root), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
