# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HiveVision reads a [Hive](https://en.wikipedia.org/wiki/Hive_(game)) board position from a single
photo and emits the placed pieces in hex coordinates. The hard requirement is **generalization**
across piece sets, surfaces, lighting, and camera angles — overfitting one setup is the explicit
failure mode. `plan.md` is the authoritative design doc; **read it before non-trivial work.** The
sibling project `../chess-vision` is the template for tooling and hard-won anti-patterns.

## Commands

```bash
uv sync                                   # create .venv and install deps
uv run pytest                             # all tests
uv run pytest tests/test_geometry.py      # one file
uv run pytest tests/test_geometry.py::test_name   # one test
uv run ruff check .                       # lint
uv run ruff format .                      # format (line-length 100)

uv run python -m hivevision.capture       # capture/label web app -> http://127.0.0.1:8000
uv run python scripts/check_lattice.py    # lattice-recovery regression on labelled photos
uv run python scripts/check_lattice.py --consistency   # group a same-position set by recovered board

# Tile detector (Phase 3). ultralytics/onnx live in the `yolo` group; torch is the plain
# CPU PyPI wheel (this is not a CUDA machine — use --device cpu, or '0' on a GPU box).
uv sync --group yolo
uv run python -m hivevision.data.yolo_pose_export   # labels.jsonl -> YOLO-pose dataset
uv run --group yolo python scripts/train_detector.py --device cpu   # builds dataset + trains
uv run --group yolo python scripts/eval_detector.py --ckpt runs/detector/weights/best.pt
```

`uv run hivevision <image>` (and `python predict.py <image>`) is a stub — the read-from-photo
pipeline (phases 3–4) is not built yet.

## Architecture

The core principle, inherited from chess-vision: **learned localization + deterministic geometry**,
never tuned heuristics. The image→position pipeline:

1. **Tile detection + class + icon-center keypoint** (Phase 3, scaffolded). A **YOLO-pose** model
   predicts, per tile, its 16-class label + icon-center keypoint in the natural un-warped photo —
   one detected object = one tile, one keypoint = the icon center. Training is gated on real data
   (only the same-position regression set exists today). See `detector.py` + `scripts/train_detector.py`.
2. **Lattice recovery** (`geometry.py:recover_lattice`, built + validated) — the deterministic
   inverse of the labelling homography. Given only coplanar icon centers, fit the homography +
   hex lattice that snaps each center to an integer axial `(q, r)`. **This replaces board-corner
   detection** — Hive has no board, the grid is implicit in the tiles.
3. **Emit position** — list of `{color, type, q, r, z}`.

Why Hive differs from chess and why it shaped this design:
- **Tiles are flat → the icon center is on the table plane**, so it's both the natural keypoint and
  an exact point on the homography plane. None of chess's hidden-base-point machinery is needed.
- **No board to localize** → recover the free-floating lattice from the point cloud instead.
- **Positions are only relative** — no natural origin like chess's a1; pick an origin + fixed axis.

### Key modules (`hivevision/`)

- `geometry.py` — axial↔plane, neighbor table, 4-point homography (DLT), project-all, and
  `recover_lattice`. The hard part: at low angles foreshortening makes a next-ring tile closer in
  pixels than a true neighbour, so **any neighbour graph built directly on the photo is
  unreliable**. `recover_lattice` therefore **estimates perspective first** (searches the two
  perspective coefficients), which leaves an affine lattice a flood-fill can label; candidates are
  refined by ICP + leave-one-out polish and chosen by reprojection residual. `LatticeFit.residual_frac`
  is the **confidence signal** — clean recoveries sit well under ~0.07.
- `pieces.py` — the taxonomy: 8 types × 2 colors = **16 classes** (locked). Codes are `<color><type>`
  e.g. `wQ`, `bA`. `z` (stack height) belongs to a *position*, never a class; phase 1 assumes `z==0`.
- `detector.py` — `TileDetector`, a YOLO-pose inference wrapper (lazy `ultralytics` import). Returns
  per-tile `Detection(code, cls, conf, xy, box)`. `data/yolo_pose_export.py` builds the training
  dataset from `labels.jsonl` — the icon-center points are the keypoints; the **box is synthesized**
  from per-image tile pitch (labels carry no box). `scripts/train_detector.py` + `eval_detector.py`
  train and report per-tile localization + class accuracy.
- `data/store.py` — `LabelStore`, rooted at `data/`. Photos land in `data/store/inbox/`, are
  EXIF-normalized into `data/store/normalized/`, labels append to `data/labels.jsonl` (keyed by
  inbox-relative `src`, last write wins). `data/` is gitignored.
- `capture/` — **FastAPI + uvicorn** web app. `app.py:create_app(root)` serves the label UI and
  `/api/recover` (live lattice preview). Static
  JS/HTML in `capture/static/`.
- `render.py` — `render_ascii` for an ASCII board dump.

### Auto-labeling flow (the labeling-throughput trick)

Enter the logical position → click ~4 tile centers with known hex coords → solve the full **8-DOF
homography by DLT** (never an affine "drag to overlap" — it smears on oblique phone shots) → project
all remaining centers automatically → nudge the few that are off. Most labels become geometric truth.

## Conventions & anti-patterns (do not relitigate — they caused real failures in chess-vision)

- **No color thresholds, no Hough lines, no hand-built lattice indexing, no reading a keypoint off a
  bounding-box edge.** Generalization comes from learned localization + deterministic geometry.
- **Never assume top-down.** Phone photos are oblique; geometry must use the full homography.
- Keep a diverse **real held-out test set** from day one; report both per-tile and whole-position
  accuracy.
- Hex coords are **axial `(q, r)`, flat-top** hexagons. Exact formulas live in `geometry.py`.

## Dependency landmines (see `pyproject.toml`)

- **`numpy<2`** is deliberate — NumPy 2.x breaks the OpenCV builds used here.
- `torch`/`torchvision` are **not** in deps yet — they arrive in Phase 3 via a cu128 index. When
  added, always pass `torch.load(weights_only=...)` explicitly.
- Python 3.12 is pinned (`.python-version`, `requires-python >=3.12,<3.13`).
