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


def _connected_adjacency(dist: np.ndarray, thresh: float) -> list[set[int]]:
    """Threshold neighbour graph, then bridge components with shortest edges.

    Tiles within ``thresh`` are linked. A strongly oblique photo can leave a
    far/gapped tile beyond the (global) threshold, splitting the graph; since a
    real Hive is always edge-connected, we then add the shortest edge between
    each pair of components until the graph is one piece — that bridging edge is
    a true neighbour across a placement gap, so its direction stays correct.
    """
    n = len(dist)
    adj: list[set[int]] = [set(np.flatnonzero(dist[i] <= thresh).tolist()) for i in range(n)]

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
    return adj


def recover_lattice(image_pts, neighbor_factor: float = 1.4) -> LatticeFit:
    """Assign integer axial coordinates to a cloud of tile-centre image points.

    The inference-time inverse of the labelling homography: given only the
    detected/labelled icon centres (a perspective view of a regular hex grid),
    recover which ``(q, r)`` each one is. Method: a connectivity-repaired
    nearest-neighbour graph -> estimate the lattice orientation from edge angles
    -> flood-fill integer coordinates by snapping each edge to one of six
    60°-spaced directions -> fit the full homography and report its residual.

    A Hive is always edge-connected, so the graph is forced to one component; the
    residual is the mean reprojection error (real hand placement is not a perfect
    lattice, so a moderate fraction-of-spacing residual is expected, not a
    topology error). ``z`` stacks are out of scope (phase-1 flat board). The
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

    adj = _connected_adjacency(dist, neighbor_factor * d_nn)

    # Lattice orientation: edge angles fold (mod 60°) onto a single phase.
    edges = [(i, j) for i in range(n) for j in adj[i]]
    ang = np.degrees(np.arctan2(*[np.array([diff[i, j, k] for i, j in edges]) for k in (1, 0)]))
    phase = _circular_mean_mod(ang, 60.0)

    # Flood-fill from the most-connected tile.
    seed = int(max(range(n), key=lambda i: len(adj[i])))
    axial: dict[int, tuple[int, int]] = {seed: (0, 0)}
    queue = deque([seed])
    while queue:
        i = queue.popleft()
        qi, ri = axial[i]
        for j in adj[i]:
            if j in axial:
                continue
            v = pts[j] - pts[i]
            theta = np.degrees(np.arctan2(v[1], v[0]))
            k = int(round((theta - phase) / 60.0)) % 6
            dq, dr = _NEIGHBOR_DELTAS_CCW[k]
            axial[j] = (qi + dq, ri + dr)
            queue.append(j)

    coords = np.array([axial[i] for i in range(n)], dtype=np.int64)
    canonical = axial_centers([tuple(c) for c in coords], size=1.0)
    H = fit_homography(canonical, pts)
    resid = np.hypot(*(project(H, canonical) - pts).T)
    return LatticeFit(
        axial=coords,
        homography=H,
        d_nn=d_nn,
        residual_px=float(resid.mean()),
        residual_frac=float(resid.mean() / d_nn),
        max_residual_px=float(resid.max()),
        n_assigned=len(axial),
        n=n,
    )
