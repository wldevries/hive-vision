"""Screenshot a page of the running capture app (Playwright, via installed Edge).

Handy for checking the label UI without eyeballing on the device. Start the app in
one terminal, then shoot:

    uv run python -m hivevision.capture
    uv run python scripts/ui_shot.py --src IMG_20260608_204537.jpg --out runs/label.png

Omit --src for the inbox page. --dpr mimics a HiDPI screen (e.g. the Surface).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--src", default=None, help="label this photo; omit for the inbox page")
    ap.add_argument("--out", default="runs/ui.png")
    ap.add_argument("--dpr", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=740)
    ap.add_argument("--wait", type=int, default=2500, help="ms to wait for async render")
    args = ap.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}/"
    if args.src:
        url += f"label?src={args.src}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.dpr,
        )
        page.goto(url)
        page.wait_for_timeout(args.wait)
        page.screenshot(path=args.out)
        browser.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
