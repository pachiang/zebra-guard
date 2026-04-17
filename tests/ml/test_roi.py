"""RoiPolygon 測試。"""

from __future__ import annotations

import pytest

from zebraguard.ml.roi import RoiPolygon


@pytest.fixture
def unit_square() -> RoiPolygon:
    return RoiPolygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)])


def test_contains_point_inside(unit_square: RoiPolygon) -> None:
    assert unit_square.contains_point(5, 5)


def test_contains_point_outside(unit_square: RoiPolygon) -> None:
    assert not unit_square.contains_point(15, 15)


def test_contains_point_on_edge(unit_square: RoiPolygon) -> None:
    assert unit_square.contains_point(0, 5)


def test_contains_point_on_vertex(unit_square: RoiPolygon) -> None:
    assert unit_square.contains_point(10, 10)


def test_distance_to_point_outside(unit_square: RoiPolygon) -> None:
    assert unit_square.distance_to_point(15, 5) == 5.0


def test_distance_to_point_inside_is_zero(unit_square: RoiPolygon) -> None:
    assert unit_square.distance_to_point(5, 5) == 0.0


def test_json_round_trip(tmp_path, unit_square: RoiPolygon) -> None:
    path = tmp_path / "roi.json"
    unit_square.save(path)
    loaded = RoiPolygon.load(path)
    assert loaded.points == unit_square.points


def test_rejects_fewer_than_three_points() -> None:
    with pytest.raises(ValueError, match="至少需要 3"):
        RoiPolygon(points=[(0, 0), (1, 1)])


def test_rejects_self_intersecting() -> None:
    # 蝴蝶結:邊會自相交
    with pytest.raises(ValueError, match="無效"):
        RoiPolygon(points=[(0, 0), (10, 10), (10, 0), (0, 10)])
