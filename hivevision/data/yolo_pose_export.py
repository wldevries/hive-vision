"""Export a YOLO-**pose** dataset (box + icon-centre keypoint) for the tile detector.

The detector's job (plan.md Phase 3): per tile, predict its **class** (one of the 16,
see ``hivevision.pieces``) and its **icon-centre keypoint**. YOLO-pose is a single model
that does exactly that — one detected object per tile, ``nc`` classes, one keypoint —
and is the pattern inherited from chess-vision (``yolo_pose_export.py`` there).

Each label row is

    cls  cx cy w h  px py pv          (all box/point coords normalized to [0, 1])

where ``(px, py)`` is the icon centre and ``pv = 2`` (labelled/visible). Our labels carry
*only* the icon centre (it is the coplanar keypoint — there is no separate box to label),
so the box is **synthesized**: a square centred on the icon, side = ``box_frac × pitch``,
where ``pitch`` is the per-image median nearest-neighbour spacing of the labelled centres
(≈ one hex cell). That box is deliberately coarse — the keypoint carries the geometry that
lattice recovery needs; the box only has to be good enough for YOLO to find the tile. At
grazing angles a square is an approximation of the foreshortened tile, which is fine here.

Input is the capture app's ``labels.jsonl`` (one position per labelled photo, points in the
EXIF-normalized image frame; see ``hivevision.data.store``). Output is a standard Ultralytics
pose dataset (``images/{train,val}``, ``labels/{train,val}``, ``data.yaml``). The normalized
JPEG is hardlinked (copied across filesystems) — no re-encode, and the pixel frame provably
matches the labels. The train/val split is a deterministic per-photo hash so it is stable
across rebuilds and never splits a single photo across both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

from hivevision.data.store import LabelStore
from hivevision.pieces import CLASS_INDEX, CLASSES, is_valid_class

# Synthetic box side as a fraction of the per-image tile pitch (centre-to-centre of
# adjacent tiles). ~1.0 ≈ one hex cell; a touch under 1 keeps neighbouring boxes from
# overlapping too heavily while still covering the tile's coloured rim.
DEFAULT_BOX_FRAC = 0.95


def _image_pitch(pts: np.ndarray) -> float:
    """Median nearest-neighbour distance among icon centres (the per-image tile pitch)."""
    if len(pts) < 2:
        return 0.0
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def pose_lines(points: list[dict], width: int, height: int, box_frac: float) -> list[str]:
    """Normalized YOLO-pose rows for one photo's labelled icon centres.

    ``points`` is ``[{label, x, y}]`` in the image frame. Unknown class codes are skipped.
    Returns ``[]`` if nothing usable (caller drops the image).
    """
    valid = [p for p in points if is_valid_class(str(p["label"]))]
    if not valid:
        return []
    xy = np.array([[float(p["x"]), float(p["y"])] for p in valid], dtype=np.float64)
    pitch = _image_pitch(xy)
    if pitch <= 0:  # single tile: fall back to a small box relative to the frame
        pitch = 0.05 * min(width, height)
    half = box_frac * pitch / 2.0

    lines = []
    for p, (x, y) in zip(valid, xy, strict=True):
        x1, y1 = max(0.0, x - half), max(0.0, y - half)
        x2, y2 = min(float(width), x + half), min(float(height), y + half)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        px = min(max(x / width, 0.0), 1.0)
        py = min(max(y / height, 0.0), 1.0)
        cls = CLASS_INDEX[str(p["label"])]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {px:.6f} {py:.6f} 2")
    return lines


def _is_val(src: str, val_frac: float, seed: int) -> bool:
    """Deterministic per-photo split: stable across rebuilds, whole photos only."""
    h = hashlib.sha1(f"{seed}:{src}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < val_frac


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:  # cross-filesystem or unsupported: fall back to a copy
        shutil.copy2(src, dst)


def build_yolo_pose_dataset(
    store_root: str | Path = Path("data"),
    out_dir: str | Path = Path("data/yolo_pose"),
    val_frac: float = 0.2,
    seed: int = 0,
    box_frac: float = DEFAULT_BOX_FRAC,
    test_prefixes: tuple[str, ...] = (),
) -> tuple[Path, dict[str, int]]:
    """Build a YOLO-pose dataset from the capture store's ``labels.jsonl``.

    ``test_prefixes`` holds photos whose ``src`` starts with any given prefix out of
    train/val entirely and into a ``test`` split — a position-level holdout (one capture
    session per prefix) that measures generalization to an unseen board, the honest test
    per plan.md. The remaining photos get the deterministic per-photo train/val hash split.

    Returns ``(data_yaml_path, counts)``. ``counts`` reports images/instances per split.
    """
    store = LabelStore(Path(store_root))
    out_dir = Path(out_dir)
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val",
                "labels/test"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    counts = {f"{s}_{k}": 0 for s in ("train", "val", "test") for k in ("images", "tiles")}
    for src, row in sorted(store.load_labels().items()):
        lines = pose_lines(row.get("points", []), row["width"], row["height"], box_frac)
        if not lines:
            continue
        if any(src.startswith(p) for p in test_prefixes):
            split = "test"
        else:
            split = "val" if _is_val(src, val_frac, seed) else "train"
        stem = src.replace("/", "_").rsplit(".", 1)[0]
        img_path = store.root / row["image"]
        if not img_path.exists():  # normalized JPEG missing (re-save the label to regenerate)
            continue
        _link_or_copy(img_path, out_dir / "images" / split / f"{stem}.jpg")
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        counts[f"{split}_images"] += 1
        counts[f"{split}_tiles"] += len(lines)

    names_block = "\n".join(f"  {i}: {code}" for i, code in enumerate(CLASSES))
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "kpt_shape: [1, 3]\n"  # one icon-centre keypoint, (x, y, visibility)
        "flip_idx: [0]\n"  # single keypoint maps to itself under horizontal flip
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path, counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", type=Path, default=Path("data"), help="capture store root")
    p.add_argument("--out-dir", type=Path, default=Path("data/yolo_pose"))
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of photos held out to val")
    p.add_argument("--seed", type=int, default=0, help="split seed (stable across rebuilds)")
    p.add_argument("--box-frac", type=float, default=DEFAULT_BOX_FRAC, help="box side / tile pitch")
    p.add_argument(
        "--test-prefixes",
        default="",
        help="comma-separated src prefixes held out as the TEST split (one capture session "
        "per prefix), e.g. 'IMG_20260609_2151'",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_prefixes = tuple(p for p in args.test_prefixes.split(",") if p)
    yaml_path, counts = build_yolo_pose_dataset(
        args.store, args.out_dir, args.val_frac, args.seed, args.box_frac, test_prefixes
    )
    print(f"wrote {yaml_path}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
