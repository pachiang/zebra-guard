"""Homography 測試。"""

from __future__ import annotations

import numpy as np
import pytest

from zebraguard.ml.homography import Homography


@pytest.fixture
def identity_hom() -> Homography:
    """1 px = 1 m 的單位對映。"""
    return Homography.from_points(
        image_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
        world_points_meters=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )


def test_identity_point(identity_hom: Homography) -> None:
    wx, wy = identity_hom.image_to_world(5, 5)
    assert wx == pytest.approx(5.0, abs=1e-6)
    assert wy == pytest.approx(5.0, abs=1e-6)


def test_identity_distance(identity_hom: Homography) -> None:
    assert identity_hom.world_distance((0, 0), (3, 4)) == pytest.approx(5.0, abs=1e-6)


def test_scaled_ten_px_per_meter() -> None:
    hom = Homography.from_points(
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        world_points_meters=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    wx, wy = hom.image_to_world(50, 50)
    assert (wx, wy) == pytest.approx((5.0, 5.0), abs=1e-6)
    assert hom.world_distance((0, 0), (30, 40)) == pytest.approx(5.0, abs=1e-6)


def test_requires_four_points() -> None:
    with pytest.raises(ValueError, match="至少需要 4"):
        Homography.from_points(
            image_points=[(0, 0), (1, 1), (2, 2)],
            world_points_meters=[(0, 0), (1, 1), (2, 2)],
        )


def test_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="數量需相等"):
        Homography.from_points(
            image_points=[(0, 0), (1, 0), (1, 1), (0, 1)],
            world_points_meters=[(0, 0), (1, 0), (1, 1)],
        )


def test_json_round_trip(identity_hom: Homography, tmp_path) -> None:
    path = tmp_path / "hom.json"
    identity_hom.save(path)
    loaded = Homography.load(path)
    np.testing.assert_allclose(loaded.matrix, identity_hom.matrix)


def test_many_points_vectorised(identity_hom: Homography) -> None:
    pts = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float64)
    out = identity_hom.image_to_world_many(pts)
    np.testing.assert_allclose(out, pts, atol=1e-6)


def test_many_points_empty(identity_hom: Homography) -> None:
    out = identity_hom.image_to_world_many(np.empty((0, 2)))
    assert out.shape == (0, 2)
