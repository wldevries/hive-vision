"""Label store round-trip: inbox listing, normalize-on-save, jsonl persistence."""

from __future__ import annotations

import numpy as np
from PIL import Image

from hivevision.data.store import LabelStore


def _write_photo(path, w=64, h=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)


def test_list_and_save_roundtrip(tmp_path):
    store = LabelStore(root=tmp_path)
    _write_photo(store.inbox_dir / "day1" / "a.jpg")

    listing = store.list_inbox()
    assert [r["src"] for r in listing] == ["day1/a.jpg"]
    assert listing[0]["labeled"] is False

    points = [{"label": "wQ", "x": 10.0, "y": 12.0}, {"label": "bA", "x": 30, "y": 8}]
    row = store.save_label("day1/a.jpg", points)
    assert row["width"] == 64 and row["height"] == 48
    assert (store.normalized_dir / "day1/a.jpg.jpg").is_file()

    reloaded = store.get_label("day1/a.jpg")
    assert [p["label"] for p in reloaded["points"]] == ["wQ", "bA"]
    assert store.list_inbox()[0]["labeled"] is True

    # Re-saving the same src replaces (not duplicates) the row.
    store.save_label("day1/a.jpg", points[:1])
    assert len(store.get_label("day1/a.jpg")["points"]) == 1


def test_inbox_path_rejects_escape(tmp_path):
    store = LabelStore(root=tmp_path)
    store.inbox_dir.mkdir(parents=True)
    try:
        store.inbox_path("../../etc/passwd")
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("expected traversal to be rejected")
