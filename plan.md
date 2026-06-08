# HiveVision — design & plan

Read a [Hive](https://en.wikipedia.org/wiki/Hive_(game)) position from a single photo and output the
board state (which pieces, where, relative to each other). This is the authoritative design doc — read it
before non-trivial work. Sibling project `../chess-vision` is the template for tooling, conventions, and
hard-won anti-patterns; we inherit its philosophy and deviate only where Hive genuinely differs.

## Goal

Photo → **position** (a list of placed pieces in hex coordinates). The hard requirement, inherited from
chess-vision, is **generalization across piece sets, surfaces, lighting, and camera angles** — not making
one setup work. Overfitting to a single domain is the explicit failure mode.

## Status (2026-06-08)

Phases 0–2 built (scaffold, geometry, capture/label app, 16-class taxonomy). **Lattice recovery
prototyped and validated** on a 9-photo set of one fixed position shot from 30–90° (a strong
regression test — same board ⇒ all should recover to one *rotation* class; reflection is a
*different* position, so equivalence is rotation-only). `recover_lattice` is now perspective-aware:
multiple candidate assignments (global-orientation flood-fill, per-seed **local-frame propagation**
that tracks the lattice angle locally instead of assuming one global angle, and per-seed patch
homography bootstrap), each refined by rectify-snap ICP + leave-one-out polish, selected by
reprojection residual. `residual_frac` is a reliable **confidence** signal (clean ≲0.07).

Current result: **6/9 recover correctly**; the other 3 are loosely-placed/branchy boards at the
**hand-click noise floor** (genuinely ambiguous from noisy points alone) — the fix there is
*precise* centres (layout-entry labelling or the trained detector), not a cleverer solver.
`scripts/check_lattice.py --consistency` groups a same-position set by recovered board as a
standing regression check. Bulletproof auto-recovery is an inference-time concern, deferred.

## What makes Hive different from chess (and why it changes the architecture)

1. **Tiles are flat and thin → the icon center is on the table plane.** In chess, the point that matters
   (the base contact point) is hidden under a tall 3D piece that leans away from the camera, so it can never
   be read off the visual center or a box edge — chess-vision spends a whole model head predicting it. In
   Hive this complication **dissolves**: the icon center *is* coplanar with the table, so it is both the
   natural visual keypoint and a geometrically exact point on the homography plane. **Label icon centers.**

2. **There is no board.** Chess inference is: detect 4 corners → homography → the 8×8 lattice is fixed and
   known. Hive has nothing to localize — the hex lattice is *implicit in the tiles themselves*. So at read
   time we detect tile centers + classes, then **recover the hex lattice from the point cloud** (fit the
   transform that snaps coplanar centers onto integer axial coordinates). This **lattice-recovery step
   replaces corner detection** and is the single biggest architectural difference. (chess-vision's
   `lattice.py` is a reference, but its job there is *indexing a known grid*; ours is *fitting a
   free-floating triangular lattice*.)

3. **Positions are only ever relative.** Hex has no natural origin/orientation like chess's a1. We pick an
   origin tile and a fixed axis convention; everything is relative.

## Scope decisions (locked 2026-06-08)

- **Base + the three expansions.** 8 piece types per color → **16 classes**: Queen `Q`, Beetle `B`,
  Grasshopper `G`, Spider `S`, Ant `A`, plus Mosquito `M`, Ladybug `L`, Pillbug `P`, each in
  white `w` / black `b`. (Per-color counts in a full set: Q×1, B×2, G×3, S×2, A×3, M×1, L×1, P×1 — counts
  don't affect classes, only legality checks later.) Expansion photos will come later; the class set is
  fixed now so the data schema and model don't churn when they arrive.
- **Flat boards for phase 1.** Assume a single-layer hive, no beetle/mosquito stacks. The position format
  reserves a `z` (stack height) field, but phase-1 data and the pipeline assume `z=0`. Stacks are a later
  phase and carry a hard limit: **you cannot see what is under a beetle from a photo** (occlusion is
  unrecoverable without move history).
- **Position = custom axial-coordinate JSON.** Each placed piece is `{color, type, q, r, z}`. Natural fit
  for reading a static photo and trivial to generate label layouts from. UHP/GameString export (for engine
  interop, e.g. Mzinga) is a possible later add-on, not the primitive — GameString is move-sequence based
  and a photo has no move history.

  ```json
  [
    {"color": "w", "type": "Q", "q": 0, "r": 0, "z": 0},
    {"color": "b", "type": "A", "q": 1, "r": 0, "z": 0}
  ]
  ```

## Hex coordinate convention

Axial coordinates `(q, r)` (cube `x=q, z=r, y=-q-r`), **flat-top** hexagons (physical Hive tiles rest on a
flat edge; pointy-top is a one-line swap if it reads better against the icons). Canonical plane position of
a tile center for hex size `s` (center-to-vertex), flat-top:

```
x = s * 3/2 * q
y = s * sqrt(3) * (r + q/2)
```

The 6 neighbor directions are the standard axial offsets. Exact formulas + neighbor table live in
`hivevision/geometry.py`, which is built and **unit-tested before any ML** (mirrors chess Phase 1).

## Architecture (image → position)

Same split chess-vision proved out: **learned localization + deterministic geometry**, never tuned
heuristics.

1. **Tile detection + classification + icon-center keypoint**, in the natural un-warped photo (chess
   "Approach A"). One model predicts, per tile: its **class** (10 classes) and its **icon-center keypoint**.
   Because the icon center is coplanar, this keypoint is far easier to learn than chess's hidden base point.
2. **Lattice recovery** (deterministic): given the detected coplanar icon centers, fit the homography +
   hex-lattice that assigns each center an integer axial coordinate, minimizing residual to a regular
   triangular lattice. Inter-tile spacing fixes scale; adjacency fixes the grid. Output: every detected
   tile → `(q, r)`.
3. **Emit position** JSON (origin = chosen by a fixed rule, e.g. first-placed/queen, or arbitrary then
   normalized). Phase 1 reports relative correctness up to rigid hex symmetry.

## Auto-labeling: enter position → fit homography → project all centers

The labeling throughput trick, lifted from chess-vision's "contact points are auto-generated geometric
truth" insight:

1. Enter the logical position (assigns each piece an axial `(q, r)`).
2. Click **~4 tile centers** in the photo whose hex coords are known from the entered position.
3. Solve the full **8-DOF homography** by DLT (4 correspondences). *Not* a similarity/affine "drag to
   overlap" — that smears on oblique phone shots; 4 clicks cost the same and handle perspective exactly.
4. **Project all remaining icon centers automatically.** User nudges only the few that are off.
5. Save → per-tile labels (icon-center keypoint + class) become training/eval truth.

This makes most labels geometric truth, with manual work bounded to a handful of clicks + occasional
nudges per photo — the review loop is QA, not a per-tile labeling chore.

## Inherited anti-patterns (from chess-vision — do not relitigate)

- **No color thresholds**, **no Hough-line detection**, **no hand-built lattice indexing**, **no reading a
  keypoint off a bounding-box edge.** All caused concrete failures in the chess prior project.
- Generalization comes from *learned* localization + *deterministic* geometry. Keep a **diverse real
  held-out test set** from day one. Report both per-tile and whole-position accuracy.
- Hive-specific corollary: the chess base-point problem does **not** apply (flat tiles), but the inverse
  trap does — **do not assume top-down**; phone photos are oblique, so geometry must use the full
  homography, never an affine shortcut.

## Tooling & reproducibility (mirror chess-vision)

- `uv` for env/deps, Python 3.12 pinned via `.python-version`. Package `hivevision/`. Pin upper bounds on
  volatile deps; landmines to pre-empt: **`numpy<2`** (breaks OpenCV) and explicit
  **`torch.load(weights_only=...)`**.
- Capture/label web app (`hivevision/capture/`, Flask) — phone photos → in-browser position entry → 4-click
  homography → auto-projected labels → nudge → store. Modeled on chess-vision's capture app.
- Flat `data/` store with a `labels.jsonl` + images; optional MinIO sync and Label Studio QA later (only if
  needed — defer the heavy infra).
- Track experiments (CSV/W&B). Record dataset hashes and train/val/test splits.

## Phase order (each phase ends with a measured number)

- **Phase 0 — Scaffold.** `uv` project, package layout, `.python-version`, CI lint/test, `predict.py` shim,
  empty `data/`/`models/`. Smoke test green.
- **Phase 1 — Geometry utility.** `geometry.py`: axial↔plane, neighbor table, homography fit (4 pts),
  project-all, and a lattice-recovery function (point cloud → axial assignment). **Unit-tested** with
  synthetic point clouds (including added noise + an oblique homography). No ML.
- **Phase 2 — Capture + auto-label app.** The web tool above. Deliverable: a labeled starter set of real
  photos with mostly-geometric truth.
- **Phase 3 — Tile detector baseline.** Keypoint+class model on the captured set (natural images). Report
  per-tile localization + class accuracy (mAP / keypoint error).
- **Phase 4 — Glue to position.** Detector → lattice recovery → position JSON. Per-photo whole-position
  accuracy on a held-out set. This is the first end-to-end number.
- **Phase 5 — Generalize.** Diversify (sets, surfaces, lighting, angles); synthetic/domain randomization if
  data-starved. Drive up the weak axis revealed by Phase 4 (chess-vision found per-piece *class accuracy*
  was the lever, not localization — watch for the analog).
- **Phase 6 — Stacks (stretch).** Add `z`/beetle handling within the occlusion limit. (Expansion *classes*
  are already in scope from day one; only stacking is deferred here.)

## Open questions (revisit, not blocking)

- Lattice recovery with sparse/early positions (2–3 tiles) is under-constrained — may need a minimum tile
  count or fall back to relative-only output.
- Origin/orientation normalization rule for the output position (queen-relative? arbitrary + canonicalized
  under hex symmetry?).
- Beetle occlusion (Phase 6): top-of-stack is visible but contents below are not — likely needs explicit
  "unknown under stack" semantics rather than a guess.
