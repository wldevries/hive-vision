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
