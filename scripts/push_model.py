"""Publish a trained detector to the blob container, with reproducibility metadata.

The label store lives in the ``hive`` container (see ``hivevision.data.blob``); trained
models go in the *same* container under a separate ``models/`` prefix::

    models/<id>/best.pt          the weights
    models/<id>/metadata.json    identity + training config + dataset + metrics + provenance

The metadata is the point: it records the **architecture**, the exact label snapshot
(sha256 of ``labels.jsonl``) and per-split source filenames the model was trained on, the
eval_detector metrics on val + the held-out test, and the code/dependency versions — enough
to compare models later (list ``models/``, read each ``metadata.json``) and to reproduce one.

    uv run --group yolo python scripts/push_model.py --run runs/detector_v2 \
        --pose-dir data/yolo_pose --test-split test --notes "hold out C; train A+B+D"

By default the model id is the run-dir name and an existing model is NOT overwritten
(pass --force). Re-runs eval on val + the named test split, so it needs the built dataset
(--pose-dir) and a GPU/CPU device (--device).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Run as `python scripts/push_model.py`, so scripts/ is sys.path[0] and this resolves; the
# explicit insert keeps it working if imported differently.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detector import evaluate  # noqa: E402

from hivevision.data.blob import DEFAULT_CONTAINER, AzureBlobBackend  # noqa: E402
from hivevision.data.store import open_store  # noqa: E402
from hivevision.pieces import CLASSES  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def _versions() -> dict:
    import numpy
    import torch

    out = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__}
    try:
        import ultralytics

        out["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return out


def _split_provenance(pose_dir: Path) -> dict:
    """Per-split image count, tile count, and the source stems actually in each split."""
    out = {}
    for split in ("train", "val", "test"):
        lbl_dir = pose_dir / "labels" / split
        if not lbl_dir.is_dir():
            continue
        stems, tiles = [], 0
        for txt in sorted(lbl_dir.glob("*.txt")):
            stems.append(txt.stem)
            tiles += sum(1 for ln in txt.read_text().splitlines() if ln.strip())
        out[split] = {"images": len(stems), "tiles": tiles, "sources": stems}
    return out


def build_metadata(args: argparse.Namespace, model_id: str, weights: bytes) -> dict:
    run = Path(args.run)
    train_args = yaml.safe_load((run / "args.yaml").read_text()) if (run / "args.yaml").exists() \
        else {}
    data_yaml = yaml.safe_load((Path(args.pose_dir) / "data.yaml").read_text())

    # epochs actually run = last row of results.csv (early stopping cuts below --epochs).
    epochs_run = None
    results = run / "results.csv"
    if results.exists():
        rows = [r for r in results.read_text().splitlines() if r.strip()]
        if len(rows) > 1:
            epochs_run = int(rows[-1].split(",")[0])

    # The canonical label snapshot the model was trained against (blob is source of truth).
    store = open_store(args.root)
    labels_bytes = store.backend.read("labels.jsonl") if store.backend else \
        (Path(args.root) / "labels.jsonl").read_bytes()
    labels = store.load_labels()

    metrics = {}
    for split in {"val", args.test_split}:
        m = evaluate(run / "weights" / args.weights_name, Path(args.pose_dir), args.conf,
                     args.imgsz, args.match_frac, args.device, split)
        metrics[split] = m

    arch = str(train_args.get("model", "")).rsplit(".", 1)[0] or "unknown"
    return {
        "id": model_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "tile-detector-pose",
        "architecture": arch,
        "imgsz": train_args.get("imgsz", args.imgsz),
        "nc": data_yaml.get("nc"),
        "classes": list(CLASSES),
        "kpt_shape": data_yaml.get("kpt_shape"),
        "keypoint": "icon-centre (one per tile)",
        "training": {
            "base_weights": train_args.get("model"),
            "epochs_requested": train_args.get("epochs"),
            "epochs_run": epochs_run,
            "patience": train_args.get("patience"),
            "batch": train_args.get("batch"),
            "optimizer": train_args.get("optimizer"),
            "lr0": train_args.get("lr0"),
            "seed": train_args.get("seed"),
            "device": train_args.get("device"),
        },
        "dataset": {
            "store_container": args.container,
            "label_count": len(labels),
            "labels_sha256": _sha256(labels_bytes) if labels_bytes else None,
            "splits": _split_provenance(Path(args.pose_dir)),
        },
        "metrics": metrics,
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            **_versions(),
        },
        "weights": {
            "blob": f"models/{model_id}/{args.weights_name}",
            "filename": args.weights_name,
            "bytes": len(weights),
            "sha256": _sha256(weights),
        },
        "notes": args.notes or None,
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=Path, required=True, help="training run dir (has weights/, args.yaml)")
    p.add_argument("--id", default=None, help="model id / blob folder (default: run dir name)")
    p.add_argument("--pose-dir", type=Path, default=Path("data/yolo_pose"))
    p.add_argument("--root", type=Path, default=Path("data"), help="local store root")
    p.add_argument("--container", default=DEFAULT_CONTAINER)
    p.add_argument("--weights-name", default="best.pt", help="weights file under run/weights/")
    p.add_argument("--test-split", default="test", choices=["test", "val", "train"],
                   help="held-out split to score alongside val")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--match-frac", type=float, default=0.5)
    p.add_argument("--device", default="0", help="'cpu' or a CUDA index like '0'")
    p.add_argument("--notes", default="", help="free-text note stored in metadata")
    p.add_argument("--force", action="store_true", help="overwrite an existing model id")
    p.add_argument("--dry-run", action="store_true", help="build + print metadata, do not upload")
    args = p.parse_args(argv)

    model_id = args.id or args.run.name
    weights_path = args.run / "weights" / args.weights_name
    if not weights_path.exists():
        p.error(f"weights not found: {weights_path}")
    weights = weights_path.read_bytes()

    backend = None if args.dry_run else AzureBlobBackend.from_env(args.container)
    meta_key = f"models/{model_id}/metadata.json"
    if backend and backend.exists(meta_key) and not args.force:
        p.error(f"model '{model_id}' already exists in blob ({meta_key}); pass --force to overwrite")

    print(f"scoring {model_id} (val + {args.test_split}) ...")
    meta = build_metadata(args, model_id, weights)
    meta_json = json.dumps(meta, indent=2).encode()

    if args.dry_run:
        print(meta_json.decode())
        return 0

    backend.write(f"models/{model_id}/{args.weights_name}", weights)
    backend.write(meta_key, meta_json)
    m_test = meta["metrics"].get(args.test_split, {})
    print(f"uploaded models/{model_id}/  ({len(weights)/1e6:.1f} MB weights + metadata.json)")
    print(f"  test[{args.test_split}]: recall {m_test.get('recall'):.3f}  "
          f"class_acc {m_test.get('class_acc'):.3f}  "
          f"kpt {m_test.get('kpt_err_px_median'):.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
