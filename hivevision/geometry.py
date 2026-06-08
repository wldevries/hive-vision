"""Hex-grid geometry and the plane<->image homography.

The deterministic half of the pipeline (see plan.md). Two responsibilities:

1. **Axial hex coordinates <-> a flat metric plane.** Hive tiles are flat-top
   hexagons; a tile at axial ``(q, r)`` has a canonical centre on the table
   plane. Because the tiles are thin, that centre is *coplanar*, so a single
   homography relates the plane to the photo exactly (the property that makes
   icon centres the right keypoint — chess-vision's hidden-base-point problem
   does not exist here).

2. **Fit / apply that homography.** Given >=4 (plane, image) correspondences
   (e.g. the user clicking known tile centres during labelling), recover the
   8-DOF homography and project every other tile centre into the photo.

Built and unit-tested before any ML. Lattice *recovery* (point cloud -> axial
assignment, the inference-time inverse) is a later phase and not here yet.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

import cv2
import numpy as np

# Axial neighbour offsets, shared by both hex orientations. Order is clockwise
# starting from "east"; index it however downstream code needs.
NEIGHBORS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)

_SQRT3 = sqrt(3.0)


def axial_to_plane(q: float, r: float, size: float = 1.0) -> tuple[float, float]:
    """Centre of the flat-top hex at axial ``(q, r)`` on the metric plane.

    ``size`` is the hex circumradius (centre to a vertex). Adjacent tile centres
    are then ``sqrt(3) * size`` apart, uniformly in all six directions.
    """
    x = size * 1.5 * q
    y = size * _SQRT3 * (r + q / 2.0)
    return x, y


def axial_centers(coords, size: float = 1.0) -> np.ndarray:
    """Plane centres for an iterable of ``(q, r)`` pairs -> ``(N, 2)`` float array."""
    return np.array([axial_to_plane(q, r, size) for q, r in coords], dtype=np.float64)


def neighbor_distance(size: float = 1.0) -> float:
    """Centre-to-centre distance between adjacent flat-top tiles."""
    return _SQRT3 * size


def fit_homography(plane_pts, image_pts) -> np.ndarray:
    """Least-squares homography mapping plane points -> image points.

    Needs >=4 correspondences (the labelling anchors). With exactly 4 it is the
    exact perspective transform; with more it is the DLT least-squares fit. Raises
    on degenerate input (too few points / collinear).
    """
    src = np.asarray(plane_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4 or dst.shape[0] != src.shape[0]:
        raise ValueError("need >=4 matched (plane, image) points")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("homography fit failed (collinear or degenerate points)")
    return H


def project(H: np.ndarray, plane_pts) -> np.ndarray:
    """Map plane points through homography ``H`` to image points -> ``(N, 2)``."""
    pts = np.asarray(plane_pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))
    return out.reshape(-1, 2)


# Axial deltas for the six neighbours, in counter-clockwise order. Index k here
# corresponds to image-space direction (lattice_phase + 60*k) degrees — see
# recover_lattice. The ordering must advance one neighbour per 60° step so cycles
# in the adjacency graph close consistently.
_NEIGHBOR_DELTAS_CCW: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)


@dataclass
class LatticeFit:
    """Result of recovering a hex lattice from a cloud of tile-centre points.

    ``axial`` is an ``(N, 2)`` int array of recovered ``(q, r)`` per input point.
    ``homography`` maps canonical plane centres (``axial_centers(axial)``) -> the
    input image points; ``residual_frac`` is the mean reprojection error as a
    fraction of tile spacing — the headline goodness-of-fit number (well under 1
    means the grid was recovered cleanly). The recovered frame is only defined up
    to a hex symmetry (rotation/reflection), which the homography absorbs.
    """

    axial: np.ndarray
    homography: np.ndarray
    d_nn: float
    residual_px: float
    residual_frac: float
    max_residual_px: float
    n_assigned: int
    n: int


def _circular_mean_mod(angles_deg: np.ndarray, period: float) -> float:
    """Circular mean of angles taken modulo ``period`` (degrees)."""
    scaled = np.deg2rad(angles_deg) * (360.0 / period)
    m = np.arctan2(np.sin(scaled).mean(), np.cos(scaled).mean())
    return float(np.rad2deg(m) * (period / 360.0))


def _bridge_components(adj: list[set[int]], dist: np.ndarray) -> None:
    """Force a single connected component by adding shortest cross-component edges
    (a Hive is always edge-connected). Mutates ``adj`` in place."""
    n = len(adj)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in adj[i]:
            parent[find(i)] = find(j)
    while len({find(i) for i in range(n)}) > 1:
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j) and (best is None or dist[i, j] < best[0]):
                    best = (dist[i, j], i, j)
        _, i, j = best
        adj[i].add(j)
        adj[j].add(i)
        parent[find(i)] = find(j)


def _global_adjacency(dist: np.ndarray, thresh: float) -> list[set[int]]:
    """Tight global-threshold neighbour graph (good for the global-orientation init)."""
    adj: list[set[int]] = [set(np.flatnonzero(d <= thresh).tolist()) for d in dist]
    _bridge_components(adj, dist)
    return adj


def _local_adjacency(pts: np.ndarray, dist: np.ndarray) -> list[set[int]]:
    """Per-tile nearest-neighbour graph (scale-adaptive), forced connected.

    The threshold is relative to each tile's *own* nearest-neighbour distance, so
    foreshortened far tiles in a steeply oblique photo stay connected where a
    single global threshold would cut them off. Components are then bridged with
    the shortest edges (a Hive is always edge-connected).
    """
    n = len(pts)
    nn = dist.min(axis=1)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in np.argsort(dist[i]):
            if dist[i, j] > 1.7 * nn[i]:
                break
            adj[i].add(int(j))
    for i in range(n):
        for j in list(adj[i]):
            adj[j].add(i)
    _bridge_components(adj, dist)
    return adj


def _plane_to_axial(xy: np.ndarray, size: float = 1.0) -> np.ndarray:
    """Invert ``axial_to_plane`` then cube-round to nearest integer hex -> (N,2) int."""
    qf = xy[:, 0] / (1.5 * size)
    rf = xy[:, 1] / (_SQRT3 * size) - qf / 2.0
    x, z = qf, rf
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
    fx = (dx > dy) & (dx > dz)
    fy = (~fx) & (dy > dz)
    rx = np.where(fx, -ry - rz, rx)
    rz = np.where(fy, -rx - ry, rz)
    return np.stack([rx, rz], axis=1).astype(np.int64)


def _edge_angle(pts: np.ndarray, i: int, j: int) -> float:
    return float(np.degrees(np.arctan2(pts[j, 1] - pts[i, 1], pts[j, 0] - pts[i, 0])))


def _bfs_assign(pts, adj, seed, theta_of):
    """Flood-fill axial coords from ``seed``; ``theta_of(i, parent)`` gives tile i's
    local lattice angle. A single ``theta_of`` ignoring its args = global orientation;
    a per-tile estimate = local-frame propagation (perspective-robust)."""
    n = len(pts)
    coord = {seed: (0, 0)}
    theta = {seed: theta_of(seed, None, None, coord)}
    queue = deque([seed])
    while queue:
        i = queue.popleft()
        ti = theta[i]
        qi, ri = coord[i]
        for j in adj[i]:
            if j in coord:
                continue
            k = int(round((_edge_angle(pts, i, j) - ti) / 60.0)) % 6
            dq, dr = _NEIGHBOR_DELTAS_CCW[k]
            coord[j] = (qi + dq, ri + dr)
            theta[j] = theta_of(j, i, k, coord)
            queue.append(j)
    return np.array([coord.get(i, (0, 0)) for i in range(n)], dtype=np.int64)


def _global_init(pts, adj):
    """Flood-fill with one global orientation (folded edge-angle mean)."""
    angs = np.array([_edge_angle(pts, i, j) for i in range(len(pts)) for j in adj[i]])
    phase = _circular_mean_mod(angs, 60.0)
    seed = int(max(range(len(pts)), key=lambda i: len(adj[i])))
    return _bfs_assign(pts, adj, seed, lambda i, p, k, c: phase)


def _propagate_init(pts, adj, seed):
    """Flood-fill carrying a per-tile local orientation, pinned by the shared edge."""
    def theta_of(i, parent, k, coord):
        if parent is None:
            return _circular_mean_mod(np.array([_edge_angle(pts, i, j) for j in adj[i]]), 60.0)
        kb = (k + 3) % 6  # reverse edge i->parent is direction kb in i's frame
        return _circular_mean_mod(np.array([_edge_angle(pts, i, parent) - 60.0 * kb]), 60.0)

    return _bfs_assign(pts, adj, seed, theta_of)


def _bootstrap_init(pts, adj, dist, s):
    """Local patch (seed + neighbours) -> homography -> rectify+snap all tiles."""
    nbrs = list(adj[s])
    if len(nbrs) < 3:
        return None
    nearest = nbrs[int(np.argmin([dist[s, j] for j in nbrs]))]
    a0 = _edge_angle(pts, s, nearest)
    coord = {s: (0, 0)}
    for j in nbrs:
        coord[j] = _NEIGHBOR_DELTAS_CCW[int(round((_edge_angle(pts, s, j) - a0) / 60.0)) % 6]
    idx = list(coord)
    if len({coord[i] for i in idx}) < len(idx):
        return None
    try:
        H = fit_homography(axial_centers([coord[i] for i in idx]), pts[idx])
    except ValueError:
        return None
    snapped = _plane_to_axial(project(np.linalg.inv(H), pts))
    return snapped if len({tuple(c) for c in snapped}) == len(snapped) else None


def _resid(pts, coords) -> float:
    canonical = axial_centers([tuple(c) for c in coords], size=1.0)
    H = fit_homography(canonical, pts)
    return float(np.hypot(*(project(H, canonical) - pts).T).mean())


def _icp(pts, coords, iters: int = 25):
    """Rectify-snap-refit until stable; return best coords or None on collision."""
    if len({tuple(c) for c in coords}) < len(coords):
        return None
    best, best_r, cur = coords, _resid(pts, coords), coords
    for _ in range(iters):
        H = fit_homography(axial_centers([tuple(c) for c in cur]), pts)
        snapped = _plane_to_axial(project(np.linalg.inv(H), pts))
        if len({tuple(c) for c in snapped}) < len(snapped):
            break
        r = _resid(pts, snapped)
        if r < best_r - 1e-9:
            best, best_r = snapped, r
        if np.array_equal(snapped, cur):
            break
        cur = snapped
    return best


def _polish(pts, coords):
    """Leave-one-out re-snap: refit without each tile, replace it in the best free cell."""
    n = len(pts)
    cur = coords.copy()
    for _ in range(6):
        improved = False
        for i in range(n):
            mask = np.ones(n, bool)
            mask[i] = False
            try:
                H = fit_homography(axial_centers([tuple(c) for c in cur[mask]]), pts[mask])
            except ValueError:
                continue
            occupied = {tuple(c) for k, c in enumerate(cur) if k != i}
            base = tuple(_plane_to_axial(project(np.linalg.inv(H), pts[i : i + 1]))[0])
            cur_r = _resid(pts, cur)
            for d in [(0, 0), *_NEIGHBOR_DELTAS_CCW]:
                c = (base[0] + d[0], base[1] + d[1])
                if c in occupied or c == tuple(cur[i]):
                    continue
                trial = cur.copy()
                trial[i] = c
                if _resid(pts, trial) < cur_r - 1e-9:
                    cur, improved = trial, True
                    break
        if not improved:
            break
    return cur


def recover_lattice(image_pts) -> LatticeFit:
    """Assign integer axial coordinates to a cloud of tile-centre image points.

    The inference-time inverse of the labelling homography: given only the
    detected/labelled icon centres (a perspective view of a regular hex grid),
    recover which ``(q, r)`` each one is. Photos can be steeply oblique (~30°),
    so a single global lattice orientation in image space is unreliable; instead
    we generate several candidate assignments and keep the one whose homography
    fits best:

    - a global-orientation flood-fill,
    - a per-seed *local-frame* flood-fill (orientation estimated locally and
      propagated, so perspective rotation is tracked rather than assumed away),
    - a per-seed local-patch homography bootstrap,

    each refined by rectify-snap ICP + a leave-one-out re-snap polish, scored by
    reprojection residual. ``residual_frac`` (mean error / tile spacing) is the
    confidence signal: clean recoveries sit well under ~0.07, so a high value
    flags a doubtful board — typically loosely-placed / branchy positions at the
    click-noise floor, where the input is genuinely ambiguous. A Hive is always
    edge-connected; ``z`` stacks are out of scope (phase-1 flat board). The
    recovered frame is only defined up to a hex symmetry, which ``H`` absorbs.
    """
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        raise ValueError("need >=3 tile centres to recover a lattice")

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    d_nn = float(np.median(dist.min(axis=1)))
    adj = _local_adjacency(pts, dist)

    # Global-orientation flood-fill needs a tight (global-threshold) graph; the
    # local-frame / bootstrap generators need the per-tile graph (perspective).
    inits = [_global_init(pts, _global_adjacency(dist, 1.4 * d_nn)), _global_init(pts, adj)]
    for seed in range(n):
        inits.append(_propagate_init(pts, adj, seed))
        boot = _bootstrap_init(pts, adj, dist, seed)
        if boot is not None:
            inits.append(boot)

    best_coords, best_r = None, None
    for ini in inits:
        refined = _icp(pts, ini)
        if refined is None:
            continue
        coords = _polish(pts, refined)
        r = _resid(pts, coords)
        if best_r is None or r < best_r:
            best_coords, best_r = coords, r
    if best_coords is None:  # pragma: no cover - every init collided (degenerate input)
        best_coords = inits[0]

    canonical = axial_centers([tuple(c) for c in best_coords], size=1.0)
    H = fit_homography(canonical, pts)
    resid = np.hypot(*(project(H, canonical) - pts).T)
    return LatticeFit(
        axial=best_coords,
        homography=H,
        d_nn=d_nn,
        residual_px=float(resid.mean()),
        residual_frac=float(resid.mean() / d_nn),
        max_residual_px=float(resid.max()),
        n_assigned=n,
        n=n,
    )
