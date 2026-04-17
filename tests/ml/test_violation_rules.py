"""Violation rules 以合成 Track 測試(不依賴真實影片或 YOLO)。"""

from __future__ import annotations

import pytest

from zebraguard.ml.homography import Homography
from zebraguard.ml.roi import RoiPolygon
from zebraguard.ml.tracking import compute_track_kinematics
from zebraguard.ml.types import (
    COCO_CAR,
    COCO_PERSON,
    Detection,
    PipelineConfig,
    Track,
)
from zebraguard.ml.violation_rules import find_violations


def _det(frame: int, sec: float, bbox, cls: int, tid: int) -> Detection:
    return Detection(
        frame_idx=frame, time_sec=sec, bbox=bbox, cls=cls, conf=0.9, track_id=tid
    )


@pytest.fixture
def identity_hom() -> Homography:
    """100×100 px → 100×100 m,1 px = 1 m。"""
    return Homography.from_points(
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        world_points_meters=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )


@pytest.fixture
def roi_center() -> RoiPolygon:
    """ROI 為畫面中央一塊 20×20 的正方形(斑馬線)。"""
    return RoiPolygon(points=[(50, 40), (70, 40), (70, 60), (50, 60)])


FPS = 10.0


def _prepare(tracks: list[Track], hom: Homography) -> None:
    for t in tracks:
        compute_track_kinematics(t, hom)


def test_violation_when_car_fast_and_close(
    identity_hom: Homography, roi_center: RoiPolygon
) -> None:
    """行人在 ROI 內,車以 108 km/h 穿過 → 違規成立。"""
    # 行人 bbox bottom-center = (60, 60),落在 ROI 邊界
    ped = Track(
        track_id=1,
        cls=COCO_PERSON,
        detections=[
            _det(f, f / FPS, (55, 50, 65, 60), COCO_PERSON, 1)
            for f in range(10, 21)
        ],
    )
    # 車輛每幀前進 3 像素 = 3 m,10 fps → 30 m/s = 108 km/h
    car_dets = []
    for f in range(5, 25):
        x = 10 + (f - 5) * 3
        car_dets.append(_det(f, f / FPS, (x, 45, x + 10, 55), COCO_CAR, 2))
    car = Track(track_id=2, cls=COCO_CAR, detections=car_dets)

    _prepare([ped, car], identity_hom)

    violations = find_violations(
        tracks=[ped, car],
        roi=roi_center,
        homography=identity_hom,
        fps=FPS,
        config=PipelineConfig(min_pedestrian_frames=3),
    )
    assert len(violations) == 1
    v = violations[0]
    assert v.pedestrian_track_id == 1
    assert v.vehicle_track_id == 2
    assert v.min_vehicle_speed_kmh > 100


def test_no_violation_when_car_stops(
    identity_hom: Homography, roi_center: RoiPolygon
) -> None:
    """行人在 ROI,車輛停在 ROI 邊緣不動 → 不算違規。"""
    ped = Track(
        track_id=1,
        cls=COCO_PERSON,
        detections=[
            _det(f, f / FPS, (55, 50, 65, 60), COCO_PERSON, 1)
            for f in range(10, 21)
        ],
    )
    # 車輛停在 (46, 45, 56, 55),bottom-center=(51, 55) 在 ROI 內,速度 0
    car = Track(
        track_id=2,
        cls=COCO_CAR,
        detections=[
            _det(f, f / FPS, (46, 45, 56, 55), COCO_CAR, 2) for f in range(5, 25)
        ],
    )
    _prepare([ped, car], identity_hom)

    violations = find_violations(
        tracks=[ped, car],
        roi=roi_center,
        homography=identity_hom,
        fps=FPS,
        config=PipelineConfig(min_pedestrian_frames=3),
    )
    assert violations == []


def test_no_violation_when_car_far(
    identity_hom: Homography, roi_center: RoiPolygon
) -> None:
    """車輛離 ROI > yield_distance_m → 不算違規。"""
    ped = Track(
        track_id=1,
        cls=COCO_PERSON,
        detections=[
            _det(f, f / FPS, (55, 50, 65, 60), COCO_PERSON, 1)
            for f in range(10, 21)
        ],
    )
    # 車輛在畫面角落附近,x 約 0-20,離 ROI 左邊 x=50 很遠
    car_dets = [
        _det(f, f / FPS, (f, 5, f + 5, 15), COCO_CAR, 2) for f in range(5, 25)
    ]
    car = Track(track_id=2, cls=COCO_CAR, detections=car_dets)
    _prepare([ped, car], identity_hom)

    violations = find_violations(
        tracks=[ped, car],
        roi=roi_center,
        homography=identity_hom,
        fps=FPS,
        config=PipelineConfig(yield_distance_m=3.0, min_pedestrian_frames=3),
    )
    assert violations == []


def test_no_violation_when_pedestrian_only_one_frame(
    identity_hom: Homography, roi_center: RoiPolygon
) -> None:
    """行人只在 ROI 出現 1 幀(< min_pedestrian_frames=3)→ 不算違規。"""
    ped = Track(
        track_id=1,
        cls=COCO_PERSON,
        detections=[_det(15, 1.5, (55, 50, 65, 60), COCO_PERSON, 1)],
    )
    car_dets = [
        _det(f, f / FPS, (50 + f, 45, 60 + f, 55), COCO_CAR, 2) for f in range(10, 20)
    ]
    car = Track(track_id=2, cls=COCO_CAR, detections=car_dets)
    _prepare([ped, car], identity_hom)

    violations = find_violations(
        tracks=[ped, car],
        roi=roi_center,
        homography=identity_hom,
        fps=FPS,
        config=PipelineConfig(min_pedestrian_frames=3),
    )
    assert violations == []


def test_no_vehicle_no_violation(
    identity_hom: Homography, roi_center: RoiPolygon
) -> None:
    """只有行人沒有車輛 → 不算違規。"""
    ped = Track(
        track_id=1,
        cls=COCO_PERSON,
        detections=[
            _det(f, f / FPS, (55, 50, 65, 60), COCO_PERSON, 1) for f in range(10, 21)
        ],
    )
    _prepare([ped], identity_hom)
    violations = find_violations(
        tracks=[ped],
        roi=roi_center,
        homography=identity_hom,
        fps=FPS,
        config=PipelineConfig(min_pedestrian_frames=3),
    )
    assert violations == []
