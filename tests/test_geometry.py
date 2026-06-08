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
