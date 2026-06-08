"""Phase 1 geometry checks: axial<->plane and the plane->image homography.

The homography tests synthesize a known perspective view of a hex layout, then
confirm that fitting from a handful of anchor correspondences reproduces every
other tile centre — the auto-labelling guarantee, validated before any data.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hivevision.geometry import (
    NEIGHBORS,
    axial_centers,
    axial_to_plane,
    fit_homography,
    neighbor_distance,
    project,
    recover_lattice,
)

# A small connected hive (axial coords) used across the homography tests.
LAYOUT = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (2, -1), (2, 0)]


def test_origin_at_zero():
    assert axial_to_plane(0, 0) == (0.0, 0.0)


def test_neighbor_distances_uniform():
    size = 1.7
    c0 = np.array(axial_to_plane(0, 0, size))
    dists = [np.linalg.norm(np.array(axial_to_plane(q, r, size)) - c0) for q, r in NEIGHBORS]
    assert np.allclose(dists, neighbor_distance(size))


def _true_homography() -> np.ndarray:
    """A deliberately non-affine (perspective) plane->image transform."""
    quad_plane = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)
    quad_image = np.array([[120, 400], [900, 360], [820, 980], [60, 900]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad_plane, quad_image)


def test_homography_exact_from_four_anchors():
    size = 1.0
    plane = axial_centers(LAYOUT, size)
    H_true = _true_homography()
    gt_image = project(H_true, plane)

    # Fit from four well-spread anchors (no three collinear); every projected
    # centre must then match ground truth.
    idx = [3, 4, 6, 7]
    H_fit = fit_homography(plane[idx], gt_image[idx])
    assert np.allclose(project(H_fit, plane), gt_image, atol=1e-6)


def test_homography_robust_to_anchor_noise():
    rng = np.random.default_rng(0)
    plane = axial_centers(LAYOUT, 1.0)
    H_true = _true_homography()
    gt_image = project(H_true, plane)

    # Use all tiles as (noisy) anchors; least-squares fit should stay close.
    noisy = gt_image + rng.normal(scale=1.5, size=gt_image.shape)
    H_fit = fit_homography(plane, noisy)
    err = np.linalg.norm(project(H_fit, plane) - gt_image, axis=1)
    assert err.mean() < 3.0


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def _check_recovery_consistent(true_coords, fit):
    """The recovered frame is only defined up to a hex symmetry, so we don't
    compare coords directly — we check the assignment is a *consistent* lattice:
    distinct coords, all tiles assigned, and adjacency preserved (true neighbours
    stay axial-distance 1 apart in the recovered frame)."""
    def hex_dist(d):  # axial (dq, dr) -> hex distance
        dq, dr = int(d[0]), int(d[1])
        return max(abs(dq), abs(dr), abs(dq + dr))

    assert fit.n_assigned == fit.n
    rec = {tuple(c) for c in fit.axial}
    assert len(rec) == fit.n  # no two tiles collapsed onto one coord
    true_arr = np.array(true_coords)
    for a in range(fit.n):
        for b in range(a + 1, fit.n):
            if hex_dist(true_arr[a] - true_arr[b]) == 1:  # true neighbours...
                assert hex_dist(fit.axial[a] - fit.axial[b]) == 1  # ...stay neighbours


def test_recover_lattice_clean_perspective():
    plane = axial_centers(LAYOUT, 1.0)
    image = project(_true_homography(), plane)
    fit = recover_lattice(image)
    assert fit.residual_frac < 0.01
    _check_recovery_consistent(LAYOUT, fit)


def test_recover_lattice_with_center_noise():
    rng = np.random.default_rng(1)
    plane = axial_centers(LAYOUT, 1.0)
    image = project(_true_homography(), plane)
    # ~3% of tile spacing of click noise on every centre.
    noisy = image + rng.normal(scale=0.03 * fit_spacing(image), size=image.shape)
    fit = recover_lattice(noisy)
    assert fit.residual_frac < 0.1
    _check_recovery_consistent(LAYOUT, fit)


def fit_spacing(image_pts):
    """Median nearest-neighbour distance of a point cloud (test helper)."""
    d = np.hypot(*(image_pts[:, None, :] - image_pts[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    return np.median(d.min(axis=1))
