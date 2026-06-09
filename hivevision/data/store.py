"""The flat label store: inbox photos in, point labels out.

Layout under the store root (``data/`` by default; gitignored)::

    data/store/inbox/<relpath>          raw phone dumps you drop in (subfolders OK)
    data/store/normalized/<relpath>.jpg EXIF-baked JPEG, written when you label
    data/labels.jsonl                   one row per labelled photo (the artifact)

**Normalize on label.** Phone photos carry an EXIF orientation flag and browsers
vs. libraries disagree on when to apply it. So on save we bake the rotation into
the pixels once, store *that* JPEG, and record every point in its frame. The app
serves the same normalized pixels for marking and the trainer will read the
stored JPEG directly — one pixel frame everywhere (mirrors chess-vision).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Decodable still-image suffixes. HEIC/HEIF is excluded — neither the browser
# <img> nor cv2 decode it without extra codecs; convert to JPEG first.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def normalize_image(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode bytes, bake in EXIF orientation; return (rgb (H,W,3) uint8, (w, h))."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(im))
    h, w = rgb.shape[:2]
    return rgb, (w, h)


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array to JPEG bytes."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


@dataclass
class LabelStore:
    """Inbox listing + label persistence rooted at ``root`` (default ``data/``)."""

    root: Path = Path("data")

    @property
    def inbox_dir(self) -> Path:
        return self.root / "store" / "inbox"

    @property
    def normalized_dir(self) -> Path:
        return self.root / "store" / "normalized"

    @property
    def thumbs_dir(self) -> Path:
        return self.root / "store" / "thumbs"

    @property
    def labels_path(self) -> Path:
        return self.root / "labels.jsonl"

    # -- inbox ------------------------------------------------------------- #

    def list_inbox(self) -> list[dict]:
        """Inbox photos as ``{src, labeled, mtime}``, newest file first.

        ``src`` is the inbox-relative POSIX path (the stable id used everywhere).
        """
        if not self.inbox_dir.exists():
            return []
        labels = self.load_labels()
        rows: list[dict] = []
        for path in self.inbox_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            src = path.relative_to(self.inbox_dir).as_posix()
            rows.append(
                {
                    "src": src,
                    "labeled": src in labels,
                    "n_points": len(labels.get(src, {}).get("points", [])),
                    "mtime": path.stat().st_mtime,
                }
            )
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return rows

    def inbox_path(self, src: str) -> Path:
        """Resolve an inbox ``src``, refusing paths that escape the inbox."""
        path = (self.inbox_dir / src).resolve()
        if self.inbox_dir.resolve() not in path.parents and path != self.inbox_dir.resolve():
            raise ValueError(f"path escapes inbox: {src!r}")
        if not path.is_file():
            raise FileNotFoundError(src)
        return path

    def normalized_bytes(self, src: str) -> bytes:
        """EXIF-normalized JPEG for display/marking (computed from the inbox file)."""
        data = self.inbox_path(src).read_bytes()
        rgb, _ = normalize_image(data)
        return encode_jpeg(rgb)

    def thumb_bytes(self, src: str, max_w: int = 320) -> bytes:
        """Small EXIF-normalized JPEG for the inbox grid — the full photo is far too
        big to send per card. Cached on disk under ``store/thumbs/`` and regenerated
        when the source photo is newer than the cache (mtime check)."""
        inbox = self.inbox_path(src)  # validates path + existence
        cache = self.thumbs_dir / f"{src}.w{max_w}.jpg"
        if cache.is_file() and cache.stat().st_mtime >= inbox.stat().st_mtime:
            return cache.read_bytes()
        rgb, (w, h) = normalize_image(inbox.read_bytes())
        if w > max_w:
            rgb = cv2.resize(rgb, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
        data = encode_jpeg(rgb, quality=80)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    # -- labels ------------------------------------------------------------ #

    def load_labels(self) -> dict[str, dict]:
        """All label rows keyed by ``src`` (last write wins)."""
        out: dict[str, dict] = {}
        if not self.labels_path.exists():
            return out
        for line in self.labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["src"]] = row
        return out

    def get_label(self, src: str) -> dict | None:
        return self.load_labels().get(src)

    def save_label(self, src: str, points: list[dict]) -> dict:
        """Normalize the inbox image, persist it, and upsert the label row.

        ``points`` is a list of ``{label, x, y}`` in the normalized image frame.
        Returns the stored row.
        """
        data = self.inbox_path(src).read_bytes()
        rgb, (w, h) = normalize_image(data)

        out_path = self.normalized_dir / f"{src}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(encode_jpeg(rgb))

        row = {
            "src": src,
            "image": out_path.relative_to(self.root).as_posix(),
            "width": w,
            "height": h,
            "points": [
                {"label": str(p["label"]), "x": float(p["x"]), "y": float(p["y"])}
                for p in points
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }

        labels = self.load_labels()
        labels[src] = row
        self._rewrite(labels)
        return row

    def _rewrite(self, labels: dict[str, dict]) -> None:
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(labels[k], ensure_ascii=False) for k in sorted(labels)]
        self.labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
