"""The flat label store: inbox photos in, point labels out.

The **canonical** store is an object store addressed by flat POSIX keys::

    inbox/<src>            raw phone dumps (subfolders OK; <src> is the stable id)
    normalized/<src>.jpg   EXIF-baked JPEG, written when you label
    labels.jsonl           one row per labelled photo (the artifact, last write wins)

With a :class:`~hivevision.data.blob.AzureBlobBackend` those keys are blobs, so you can
label from several computers against one container. The local ``root`` (``data/`` by
default; gitignored) is then a **read-through cache**: images land there on first read and
on save, so repeated reads and training stay on local disk, while ``labels.jsonl`` is always
re-read from the backend before a save so machines don't clobber each other. With
``backend=None`` the store is purely local and ``root`` *is* the source of truth — the
original single-machine behaviour, used by the tests. Local cache layout mirrors the keys::

    <root>/store/inbox/<relpath>           <- inbox/<relpath>
    <root>/store/normalized/<relpath>.jpg  <- normalized/<relpath>.jpg
    <root>/store/thumbs/<relpath>.w<W>.jpg   (always local; regenerable)
    <root>/labels.jsonl                    <- labels.jsonl

**Normalize on label.** Phone photos carry an EXIF orientation flag and browsers
vs. libraries disagree on when to apply it. So on save we bake the rotation into
the pixels once, store *that* JPEG, and record every point in its frame. The app
serves the same normalized pixels for marking and the trainer will read the
stored JPEG directly — one pixel frame everywhere (mirrors chess-vision).
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageOps

from hivevision.data.blob import CONNECTION_STRING_ENV, AzureBlobBackend, StorageBackend

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


def inbox_key(src: str) -> str:
    return f"inbox/{src}"


def normalized_key(src: str) -> str:
    return f"normalized/{src}.jpg"


@dataclass
class LabelStore:
    """Inbox listing + label persistence over ``backend`` (or local ``root`` if None)."""

    root: Path = field(default_factory=lambda: Path("data"))
    backend: StorageBackend | None = None

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

    # -- key <-> local-cache path + storage primitives --------------------- #

    def _local(self, key: str) -> Path:
        """Local cache path for a storage ``key``."""
        return self.root / key if key == "labels.jsonl" else self.root / "store" / key

    def _safe_src(self, src: str) -> str:
        """Reject ``src`` values that would escape the inbox (absolute or ``..``)."""
        p = PurePosixPath(src)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"path escapes inbox: {src!r}")
        return p.as_posix()

    def _read(self, key: str) -> bytes | None:
        """Bytes for ``key``, read-through cached to local disk; ``None`` if absent.

        ``labels.jsonl`` is intentionally *not* served from cache (see :meth:`load_labels`).
        """
        local = self._local(key)
        if self.backend is None:
            return local.read_bytes() if local.is_file() else None
        if local.is_file():
            return local.read_bytes()
        data = self.backend.read(key)
        if data is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        return data

    def _write(self, key: str, data: bytes) -> None:
        """Write ``key`` to the canonical backend (if any) and the local cache."""
        local = self._local(key)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        if self.backend is not None:
            self.backend.write(key, data)

    def ensure_local(self, key: str) -> Path | None:
        """Materialize ``key`` on local disk and return its path (``None`` if absent).

        Used by training/inspection, which want a real file path rather than bytes.
        """
        local = self._local(key)
        if local.is_file():
            return local
        if self._read(key) is None:
            return None
        return local

    def normalized_path(self, src: str) -> Path | None:
        """Local path to the normalized JPEG for ``src`` (downloaded on demand)."""
        return self.ensure_local(normalized_key(self._safe_src(src)))

    # -- inbox ------------------------------------------------------------- #

    def list_inbox(self) -> list[dict]:
        """Inbox photos as ``{src, labeled, n_points, mtime}``, newest first.

        ``src`` is the inbox-relative POSIX path (the stable id used everywhere).
        """
        labels = self.load_labels()
        rows: list[dict] = []
        if self.backend is not None:
            for name, mtime in self.backend.list("inbox/"):
                src = name[len("inbox/") :]
                if not src or PurePosixPath(src).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                rows.append(self._inbox_row(src, mtime, labels))
        elif self.inbox_dir.exists():
            for path in self.inbox_dir.rglob("*"):
                if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                    continue
                src = path.relative_to(self.inbox_dir).as_posix()
                rows.append(self._inbox_row(src, path.stat().st_mtime, labels))
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return rows

    @staticmethod
    def _inbox_row(src: str, mtime: float, labels: dict[str, dict]) -> dict:
        return {
            "src": src,
            "labeled": src in labels,
            "n_points": len(labels.get(src, {}).get("points", [])),
            "mtime": mtime,
        }

    def inbox_path(self, src: str) -> Path:
        """Resolve an inbox ``src`` to a local file path (downloaded on demand)."""
        src = self._safe_src(src)
        path = self.ensure_local(inbox_key(src))
        if path is None:
            raise FileNotFoundError(src)
        return path

    def normalized_bytes(self, src: str) -> bytes:
        """EXIF-normalized JPEG for display/marking (computed from the inbox photo)."""
        data = self._read(inbox_key(self._safe_src(src)))
        if data is None:
            raise FileNotFoundError(src)
        rgb, _ = normalize_image(data)
        return encode_jpeg(rgb)

    def thumb_bytes(self, src: str, max_w: int = 320) -> bytes:
        """Small EXIF-normalized JPEG for the inbox grid — the full photo is far too
        big to send per card. Cached on disk under ``store/thumbs/``. Inbox photos are
        immutable once stored, so a cached thumb is reused as-is when the store is
        backed by blob; for a purely-local store it is regenerated when the source
        photo is newer than the cache (mtime check)."""
        src = self._safe_src(src)
        cache = self.thumbs_dir / f"{src}.w{max_w}.jpg"
        if cache.is_file() and self._thumb_fresh(cache, src):
            return cache.read_bytes()
        data = self._read(inbox_key(src))
        if data is None:
            raise FileNotFoundError(src)
        rgb, (w, h) = normalize_image(data)
        if w > max_w:
            rgb = cv2.resize(rgb, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
        out = encode_jpeg(rgb, quality=80)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(out)
        return out

    def _thumb_fresh(self, cache: Path, src: str) -> bool:
        if self.backend is not None:
            return True  # inbox blobs are immutable
        inbox = self._local(inbox_key(src))
        return inbox.is_file() and cache.stat().st_mtime >= inbox.stat().st_mtime

    # -- labels ------------------------------------------------------------ #

    def load_labels(self) -> dict[str, dict]:
        """All label rows keyed by ``src`` (last write wins).

        Read straight from the canonical backend every call (never the local cache) so a
        save on another machine is seen before we rewrite. The fetched copy is mirrored to
        the local ``labels.jsonl`` for offline inspection.
        """
        if self.backend is not None:
            data = self.backend.read("labels.jsonl")
            if data is None:
                return {}
            local = self._local("labels.jsonl")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            text = data.decode("utf-8")
        else:
            if not self.labels_path.exists():
                return {}
            text = self.labels_path.read_text(encoding="utf-8")
        out: dict[str, dict] = {}
        for line in text.splitlines():
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
        src = self._safe_src(src)
        data = self._read(inbox_key(src))
        if data is None:
            raise FileNotFoundError(src)
        rgb, (w, h) = normalize_image(data)

        self._write(normalized_key(src), encode_jpeg(rgb))

        row = {
            "src": src,
            "image": normalized_key(src),
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
        body = "\n".join(json.dumps(labels[k], ensure_ascii=False) for k in sorted(labels)) + "\n"
        self._write("labels.jsonl", body.encode("utf-8"))


def open_store(root: str | Path = Path("data")) -> LabelStore:
    """Open the store, using the Azure backend iff ``STORAGE_CONNECTION_STRING`` is set.

    Entry points (the capture app, the scripts) call ``load_dotenv()`` before this, so the
    variable is read from ``.env``. Tests construct ``LabelStore`` directly and never set the
    variable, so they stay purely local — this never reaches out to the network for them.
    """
    backend = AzureBlobBackend.from_env() if os.environ.get(CONNECTION_STRING_ENV) else None
    return LabelStore(root=Path(root), backend=backend)
