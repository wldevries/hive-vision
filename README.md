# HiveVision

Read a [Hive](https://en.wikipedia.org/wiki/Hive_(game)) position from a photo and output the board
state (which pieces, where, in hex coordinates), designed to generalize across piece sets, surfaces,
and environments. See [`plan.md`](plan.md) for the design and current state.

## Approach (image → position)

1. **Tile detection + classification + icon-center keypoint** in the natural photo. Hive tiles are
   flat, so the icon center is on the table plane — it's both the natural keypoint and a geometrically
   exact point on the homography plane (no hidden-base-point problem like chess).
2. **Lattice recovery** (deterministic) — fit the homography/hex-lattice that snaps the detected
   coplanar centers onto integer axial coordinates. This replaces board-corner detection: Hive has no
   board, so the grid is implicit in the tiles themselves.
3. **Emit position** — a list of `{color, type, q, r, z}` pieces.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via `.python-version`.

```bash
uv sync          # create .venv and install deps
uv run pytest    # run tests
```

## Labeling photos

Drop phone photos into `data/store/inbox/` (subfolders OK), then label them in the web app:

```bash
uv run python -m hivevision.capture        # open http://127.0.0.1:8000
```

Photos are EXIF-normalized on save; labels are written to `data/labels.jsonl`. `data/` is gitignored.

## Status

Phases 0–2 (scaffold, geometry, capture/label app). The read-from-photo CLI is not implemented yet
(phases 3–4). See `plan.md`.
