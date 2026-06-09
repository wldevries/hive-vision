"""Train the YOLO-pose tile detector (box + class + icon-centre keypoint).

The learned half of the pipeline (plan.md Phase 3). One model predicts, per tile, its class
(16) and icon-centre keypoint; downstream, the centres feed lattice recovery (Phase 4).

    uv sync --group yolo
    uv run --group yolo python scripts/train_detector.py \
        --model yolo11n-pose.pt --epochs 200 --imgsz 1280 --batch -1 --device 0

The dataset is auto-built from the capture store's labels.jsonl into --pose-dir on first run
(normalized JPEGs hardlinked, boxes synthesized from the icon centres — see
hivevision/data/yolo_pose_export.py); pass --rebuild to regenerate. Use --device cpu on a
machine without CUDA (slow but fine for a smoke run on a few photos).

Notes:
  - YOLO-pose trains box + keypoint jointly; the keypoint (OKS) loss is what localization
    needs. The class head is the likely weak axis (plan.md: chess found class accuracy was the
    lever, not localization) — watch per-class accuracy once there is real data.
  - --imgsz 1280: tile icons are small in phone photos; this is the main accuracy/speed lever.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Ultralytics imports polars to write results.csv; on Windows-ARM64 polars' CPU-feature
# pre-flight check wrongly tests for the x86 'sse3' flag and aborts ("unknown feature flag:
# 'sse3'"), even though the native arm64 binary runs fine. Skip the bogus check. Harmless on
# x86 (the check would pass). Must be set before ultralytics imports polars.
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from hivevision.data.yolo_pose_export import build_yolo_pose_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--store", type=Path, default=Path("data"), help="capture store root (labels.jsonl)")
    add("--pose-dir", type=Path, default=Path("data/yolo_pose"), help="built dataset dir")
    add("--rebuild", action="store_true", help="regenerate the dataset even if it exists")
    add("--val-frac", type=float, default=0.2, help="fraction of photos held out to val")
    add("--seed", type=int, default=0)
    add("--model", default="yolo11n-pose.pt", help="Ultralytics pose model (yolo11n/s-pose .pt)")
    add("--epochs", type=int, default=200)
    add("--imgsz", type=int, default=1280, help="train/val image size (multiple of 32)")
    add("--batch", type=int, default=-1, help="batch size (-1 = Ultralytics auto)")
    add("--device", default="cpu", help="'cpu', or a CUDA index like '0'")
    add("--workers", type=int, default=8)
    add("--patience", type=int, default=10, help="early-stop patience (epochs without val gain)")
    add("--project", type=Path, default=Path("runs"))
    add("--name", default="detector", help="run name under --project")
    add("--resume", action="store_true", help="resume an interrupted run from its last.pt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.resume:
        from ultralytics import YOLO

        last = args.project.resolve() / args.name / "weights" / "last.pt"
        print(f"resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        print(f"done. best weights: {Path(model.trainer.best)}")
        return 0

    yaml_path = args.pose_dir / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        print(f"building YOLO-pose dataset -> {args.pose_dir}")
        yaml_path, counts = build_yolo_pose_dataset(
            args.store, args.pose_dir, args.val_frac, args.seed
        )
        print(f"  {yaml_path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    else:
        print(f"using existing dataset {yaml_path} (pass --rebuild to regenerate)")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=str(args.project.resolve()),  # absolute: avoid Ultralytics' runs/ nesting
        name=args.name,
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    print(f"done. best weights: {best}")
    print(f"eval:  uv run --group yolo python scripts/eval_detector.py --ckpt {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
