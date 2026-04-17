"""aggregate_tracks 與 compute_track_kinematics 測試。"""

from __future__ import annotations

import pytest

from zebraguard.ml.homography import Homography
from zebraguard.ml.tracking import aggregate_tracks, compute_track_kinematics
from zebraguard.ml.types import COCO_CAR, COCO_PERSON, Detection, Track


def _det(frame: int, sec: float, bbox, cls: int, tid: int | None) -> Detection:
    return Detection(
        frame_idx=frame, time_sec=sec, bbox=bbox, cls=cls, conf=0.9, track_id=tid
    )


@pytest.fixture
def identity_hom() -> Homography:
    return Homography.from_points(
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        world_points_meters=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )


def test_aggregate_groups_by_track_id() -> None:
    dets = [
        _det(0, 0.0, (0, 0, 10, 10), COCO_PERSON, 1),
        _det(1, 0.1, (5, 0, 15, 10), COCO_PERSON, 1),
        _det(0, 0.0, (50, 0, 60, 10), COCO_CAR, 2),
    ]
    tracks = aggregate_tracks(dets)
    assert len(tracks) == 2
    t1 = next(t for t in tracks if t.track_id == 1)
    assert len(t1.detections) == 2
    assert t1.cls == COCO_PERSON


def test_aggregate_drops_untracked() -> None:
    dets = [_det(0, 0.0, (0, 0, 10, 10), COCO_PERSON, None)]
    assert aggregate_tracks(dets) == []


def test_aggregate_sorts_by_frame() -> None:
    dets = [
        _det(5, 0.5, (0, 0, 10, 10), COCO_PERSON, 1),
        _det(1, 0.1, (0, 0, 10, 10), COCO_PERSON, 1),
        _det(3, 0.3, (0, 0, 10, 10), COCO_PERSON, 1),
    ]
    (t,) = aggregate_tracks(dets)
    assert [d.frame_idx for d in t.detections] == [1, 3, 5]


def test_kinematics_first_detection_none_speed(identity_hom: Homography) -> None:
    track = Track(
        track_id=1,
        cls=COCO_CAR,
        detections=[
            _det(0, 0.0, (0, 0, 10, 10), COCO_CAR, 1),
            _det(10, 1.0, (10, 0, 20, 10), COCO_CAR, 1),
        ],
    )
    compute_track_kinematics(track, identity_hom)
    # 第一幀無歷史 → None(而非 0.0,避免與真實停止混淆)
    assert track.detections[0].speed_kmh is None
    # bottom-center 從 (5, 10) 到 (15, 10):10 m / 1 s = 10 m/s = 36 km/h
    assert track.detections[1].speed_kmh == pytest.approx(36.0, abs=1e-6)


def test_kinematics_fills_world_pos(identity_hom: Homography) -> None:
    track = Track(
        track_id=1,
        cls=COCO_CAR,
        detections=[_det(0, 0.0, (10, 20, 30, 40), COCO_CAR, 1)],
    )
    compute_track_kinematics(track, identity_hom)
    # bottom-center = (20, 40);identity 映射 → (20, 40)
    assert track.detections[0].world_pos == pytest.approx((20.0, 40.0), abs=1e-6)
