"""YOLO-pose export: icon-centre labels -> a valid Ultralytics pose dataset."""

from __future__ import annotations

import numpy as np
from PIL import Image

from hivevision.data.store import LabelStore
from hivevision.data.yolo_pose_export import build_yolo_pose_dataset, pose_lines
from hivevision.pieces import CLASS_INDEX


def test_pose_lines_format_and_box():
    # Three tiles ~100px apart -> pitch 100, default box side 95px (0.475 of 200-wide frame).
    pts = [
        {"label": "wQ", "x": 100.0, "y": 100.0},
        {"label": "bA", "x": 200.0, "y": 100.0},
        {"label": "wG", "x": 100.0, "y": 200.0},
    ]
    lines = pose_lines(pts, width=400, height=400, box_frac=0.95)
    assert len(lines) == 3

    cls, cx, cy, bw, bh, px, py, pv = lines[0].split()
    assert int(cls) == CLASS_INDEX["wQ"]
    assert pv == "2"
    # keypoint is the icon centre (100/400 = 0.25); box ~95px wide (0.2375 of 400)
    assert abs(float(px) - 0.25) < 1e-6 and abs(float(py) - 0.25) < 1e-6
    assert abs(float(bw) - 0.2375) < 1e-3 and abs(float(bh) - 0.2375) < 1e-3
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, bw, bh, px, py))


def test_pose_lines_skips_unknown_classes():
    pts = [{"label": "wQ", "x": 10, "y": 10}, {"label": "ZZ", "x": 50, "y": 50}]
    lines = pose_lines(pts, width=100, height=100, box_frac=0.95)
    assert len(lines) == 1 and lines[0].startswith(f"{CLASS_INDEX['wQ']} ")


def test_build_dataset_writes_yaml_and_labels(tmp_path):
    store = LabelStore(root=tmp_path)
    img = store.inbox_dir / "a.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(img)
    store.save_label(
        "a.jpg",
        [
            {"label": "wQ", "x": 100, "y": 100},
            {"label": "bA", "x": 200, "y": 100},
        ],
    )

    out = tmp_path / "yolo_pose"
    yaml_path, counts = build_yolo_pose_dataset(tmp_path, out, val_frac=0.0)

    assert yaml_path.is_file()
    text = yaml_path.read_text(encoding="utf-8")
    assert "kpt_shape: [1, 3]" in text and "nc: 16" in text
    assert counts["train_images"] == 1 and counts["train_tiles"] == 2
    assert (out / "images" / "train" / "a.jpg").is_file()
    assert (out / "labels" / "train" / "a.txt").read_text().strip().count("\n") == 1
