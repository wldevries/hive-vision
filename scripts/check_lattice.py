"""Run lattice recovery on labelled photos and write inspection overlays.

    uv run python scripts/check_lattice.py            # all labelled photos
    uv run python scripts/check_lattice.py --src IMG_...jpg

For each photo it recovers axial (q, r) from the labelled icon centres, prints
the reprojection residual, and writes an overlay PNG (recovered coords + the
homography-reprojected grid) under data/_lattice_check/ for eyeballing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from hivevision.data.store import LabelStore, open_store
from hivevision.geometry import axial_centers, project, recover_lattice
from hivevision.render import render_ascii


def load_rows(store: LabelStore) -> list[dict]:
    return list(store.load_labels().values())


def overlay(img: np.ndarray, pts: np.ndarray, labels: list[str], fit) -> np.ndarray:
    out = img.copy()
    # reprojected canonical centres (green) — should land on the labelled dots
    reproj = project(fit.homography, axial_centers([tuple(c) for c in fit.axial], size=1.0))
    for (rx, ry) in reproj:
        cv2.drawMarker(out, (int(rx), int(ry)), (60, 220, 60), cv2.MARKER_CROSS, 26, 3)
    for i, (x, y) in enumerate(pts):
        q, r = fit.axial[i]
        cv2.circle(out, (int(x), int(y)), 16, (255, 90, 40), 3)
        cv2.putText(out, f"{labels[i]} ({q},{r})", (int(x) + 20, int(y) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, f"{labels[i]} ({q},{r})", (int(x) + 20, int(y) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 220, 255), 2, cv2.LINE_AA)
    return out


def _canon_rotation(coords) -> tuple:
    """Canonical form of a board shape under the 6 hex rotations (no reflection).

    Lets us check whether several photos of the *same* physical position recover
    to the same board — a far stronger correctness test than residual alone.
    """
    cur = [(int(q), -int(q) - int(r), int(r)) for q, r in coords]  # axial -> cube
    best = None
    for _ in range(6):
        cur = [(-z, -x, -y) for x, y, z in cur]  # rotate 60°
        ax = [(c[0], c[2]) for c in cur]
        mq = min(q for q, _ in ax)
        mr = min(r for _, r in ax)
        key = tuple(sorted((q - mq, r - mr) for q, r in ax))
        if best is None or key < best:
            best = key
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--src", default=None, help="only this photo (else all labelled)")
    ap.add_argument(
        "--consistency",
        action="store_true",
        help="group all labelled photos by recovered board (rotation-only); use when a set is "
        "known to be the same physical position to spot recovery errors",
    )
    args = ap.parse_args(argv)

    load_dotenv()  # picks up STORAGE_CONNECTION_STRING so open_store() reads from blob
    store = open_store(args.root)
    rows = load_rows(store)
    if args.src:
        rows = [r for r in rows if r["src"] == args.src]
    if not rows:
        print("no matching labelled photos")
        return 1

    if args.consistency:
        groups: dict[tuple, list] = {}
        for row in rows:
            pts = np.array([[p["x"], p["y"]] for p in row["points"]], dtype=np.float64)
            if len(pts) < 3:
                continue
            fit = recover_lattice(pts)
            groups.setdefault(_canon_rotation(fit.axial), []).append(
                (row["src"], round(fit.residual_frac * 100, 1))
            )
        print(f"{len(rows)} photos -> {len(groups)} distinct boards (rotation-only):")
        for i, members in enumerate(sorted(groups.values(), key=len, reverse=True), 1):
            print(f"  board #{i} ({len(members)}): {members}")
        return 0

    out_dir = Path("runs") / "lattice_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        pts = np.array([[p["x"], p["y"]] for p in row["points"]], dtype=np.float64)
        labels = [p["label"] for p in row["points"]]
        if len(pts) < 3:
            print(f"{row['src']}: only {len(pts)} points, skipping")
            continue
        fit = recover_lattice(pts)
        print(
            f"\n{row['src']}: n={fit.n} assigned={fit.n_assigned} "
            f"d_nn={fit.d_nn:.1f}px  residual={fit.residual_px:.1f}px "
            f"({fit.residual_frac * 100:.1f}% of spacing)  max={fit.max_residual_px:.1f}px"
        )
        placements = [(int(c[0]), int(c[1]), labels[i]) for i, c in enumerate(fit.axial)]
        print(render_ascii(placements))

        img_path = store.normalized_path(row["src"])
        img = cv2.imread(str(img_path)) if img_path else None
        if img is None:
            print(f"  (no normalized image for {row['src']}, skipping overlay)")
            continue
        out = overlay(img, pts, labels, fit)
        scale = 1600 / out.shape[1]
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dst = out_dir / (row["src"].replace("/", "_") + ".lattice.png")
        cv2.imwrite(str(dst), out)
        print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
