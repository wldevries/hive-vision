"""Evaluate the tile detector on the held-out val split — the Phase 3 deliverable number.

Reports the two things plan.md Phase 3 asks for, per tile:
  - **localization**: icon-centre keypoint error (px, and as a fraction of the tile pitch)
  - **class accuracy**: fraction of matched tiles whose predicted class is correct
plus detection recall / precision (a detection matches a GT tile if its keypoint is the
nearest unused one within --match-frac × pitch).

    uv run --group yolo python scripts/eval_detector.py --ckpt runs/detector/weights/best.pt

Reads the built dataset's val split (images/val + labels/val under --pose-dir), so run after
the dataset has been built (train_detector.py builds it; or `python -m
hivevision.data.yolo_pose_export`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hivevision.detector import TileDetector


def _load_gt(label_path: Path, w: int, h: int) -> tuple[np.ndarray, list[int]]:
    """GT icon centres (px) and class ids from one YOLO-pose label file."""
    xy, cls = [], []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        f = ln.split()
        if len(f) < 8:
            continue
        cls.append(int(f[0]))
        xy.append([float(f[5]) * w, float(f[6]) * h])
    return np.array(xy, dtype=np.float64).reshape(-1, 2), cls


def _pitch(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def evaluate(
    ckpt: Path, pose_dir: Path, conf: float, imgsz: int, match_frac: float, device: str | None,
    split: str = "val",
) -> dict:
    det = TileDetector(ckpt, device=device)
    img_dir, lbl_dir = pose_dir / "images" / split, pose_dir / "labels" / split
    images = sorted(p for p in img_dir.glob("*.jpg"))

    n_gt = n_det = n_match = n_cls_ok = 0
    errs_px, errs_frac = [], []
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        gt_xy, gt_cls = _load_gt(lbl_path, w, h)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = det.detect(rgb, conf=conf, imgsz=imgsz)
        n_gt += len(gt_xy)
        n_det += len(dets)

        pitch = _pitch(gt_xy) or (0.05 * min(w, h))
        thresh = match_frac * pitch
        det_xy = np.array([d.xy for d in dets], dtype=np.float64).reshape(-1, 2)
        used = set()
        for gi, g in enumerate(gt_xy):  # greedy nearest match, GT-first
            if not len(det_xy):
                break
            dist = np.hypot(*(det_xy - g).T)
            for di in np.argsort(dist):
                if di in used or dist[di] > thresh:
                    if di in used:
                        continue
                    break
                used.add(int(di))
                n_match += 1
                errs_px.append(float(dist[di]))
                errs_frac.append(float(dist[di] / pitch))
                if dets[di].cls == gt_cls[gi]:
                    n_cls_ok += 1
                break

    errs_px = np.array(errs_px) if errs_px else np.array([np.nan])
    errs_frac = np.array(errs_frac) if errs_frac else np.array([np.nan])
    return {
        "images": len(images),
        "gt_tiles": n_gt,
        "detections": n_det,
        "matched": n_match,
        "recall": n_match / n_gt if n_gt else float("nan"),
        "precision": n_match / n_det if n_det else float("nan"),
        "class_acc": n_cls_ok / n_match if n_match else float("nan"),
        "kpt_err_px_mean": float(np.nanmean(errs_px)),
        "kpt_err_px_median": float(np.nanmedian(errs_px)),
        "kpt_err_frac_mean": float(np.nanmean(errs_frac)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True, help="trained best.pt")
    p.add_argument("--pose-dir", type=Path, default=Path("data/yolo_pose"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--match-frac", type=float, default=0.5, help="match radius / tile pitch")
    p.add_argument("--device", default="cpu", help="'cpu', or a CUDA index like '0'")
    p.add_argument("--split", default="val", choices=["val", "test", "train"], help="split scored")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    m = evaluate(args.ckpt, args.pose_dir, args.conf, args.imgsz, args.match_frac, args.device,
                 args.split)
    print(f"{args.split} images: {m['images']}  ·  GT tiles: {m['gt_tiles']}")
    print(f"  recall    {m['recall']:.3f}   precision {m['precision']:.3f}")
    print(f"  class_acc {m['class_acc']:.3f}   (over {m['matched']} matched tiles)")
    print(
        f"  kpt err   {m['kpt_err_px_median']:.1f}px median, "
        f"{m['kpt_err_px_mean']:.1f}px mean  ({m['kpt_err_frac_mean']:.3f} of pitch)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
