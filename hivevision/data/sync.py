"""Push local store files up to the canonical blob container.

The store reads through the blob on demand, but two things still originate on a local
machine and have to be *pushed*: the existing data you already labelled (a one-time
migration) and any new phone photos you drop into the inbox.

    uv run python -m hivevision.data.sync                 # upload new inbox + normalized
    uv run python -m hivevision.data.sync --migrate       # also push local labels.jsonl
    uv run python -m hivevision.data.sync --force         # re-upload even if the blob exists

Default is skip-existing, so it is safe to run repeatedly to upload freshly dropped photos.
``--migrate`` additionally uploads ``labels.jsonl`` — run it **once**, from the machine that
holds your existing labels, to seed the container. Don't push ``labels.jsonl`` routinely: the
capture app writes labels straight to the blob, so the local copy is only a cache and pushing
a stale one would clobber newer work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from hivevision.data.blob import DEFAULT_CONTAINER, AzureBlobBackend, StorageBackend
from hivevision.data.store import IMAGE_SUFFIXES


def _push_tree(backend: StorageBackend, base: Path, prefix: str, force: bool) -> tuple[int, int]:
    """Upload every image under ``base`` as ``{prefix}{relpath}``. Returns (uploaded, skipped)."""
    uploaded = skipped = 0
    if not base.exists():
        return 0, 0
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = prefix + path.relative_to(base).as_posix()
        if not force and backend.exists(key):
            skipped += 1
            continue
        backend.write(key, path.read_bytes())
        print(f"  uploaded {key}")
        uploaded += 1
    return uploaded, skipped


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("data"), help="local store root")
    ap.add_argument("--container", default=DEFAULT_CONTAINER, help="blob container name")
    ap.add_argument("--force", action="store_true", help="re-upload even if the blob exists")
    ap.add_argument(
        "--migrate", action="store_true", help="also push local labels.jsonl (one-time seed)"
    )
    args = ap.parse_args(argv)

    backend = AzureBlobBackend.from_env(args.container)

    print("inbox:")
    up_i, sk_i = _push_tree(backend, args.root / "store" / "inbox", "inbox/", args.force)
    print(f"  {up_i} uploaded, {sk_i} already present")

    print("normalized:")
    up_n, sk_n = _push_tree(backend, args.root / "store" / "normalized", "normalized/", args.force)
    print(f"  {up_n} uploaded, {sk_n} already present")

    if args.migrate:
        labels = args.root / "labels.jsonl"
        if labels.is_file():
            backend.write("labels.jsonl", labels.read_bytes())
            print("labels.jsonl uploaded")
        else:
            print("labels.jsonl: nothing local to migrate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
