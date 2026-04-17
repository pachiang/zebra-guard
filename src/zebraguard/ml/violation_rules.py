"""違規判定規則引擎。

規則(對照 docs/legal-rules.md § 3):
  A. 行人 bbox 底邊中心在 ROI 內,且連續 ≥ min_pedestrian_frames 幀
  B. 同一時段內,車輛 bbox 底邊中心到 ROI 的**世界距離** ≤ yield_distance_m
  C. 車輛在「最接近時刻」前 window_before_sec 秒內,
     最低速度 > stop_threshold_kmh(亦即未曾停讓)

三條件全部成立時,產生一筆 ViolationEvent。使用者仍須在 UI 審查。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from zebraguard.ml.homography import Homography
from zebraguard.ml.roi import RoiPolygon
from zebraguard.ml.types import Detection, PipelineConfig, Track, ViolationEvent


@dataclass(slots=True)
class _PedestrianInterval:
    track_id: int
    start_frame: int  # inclusive
    end_frame: int  # inclusive


def _find_pedestrian_intervals(
    track: Track, roi: RoiPolygon, min_frames: int
) -> list[_PedestrianInterval]:
    """找出連續 ≥ min_frames 幀位於 ROI 內的時段。"""
    intervals: list[_PedestrianInterval] = []
    run_start_idx: int | None = None
    run_start_frame: int | None = None

    def _close_run(end_idx: int, end_frame: int) -> None:
        assert run_start_idx is not None
        if (end_idx - run_start_idx + 1) >= min_frames:
            intervals.append(
                _PedestrianInterval(
                    track_id=track.track_id,
                    start_frame=run_start_frame,  # type: ignore[arg-type]
                    end_frame=end_frame,
                )
            )

    for i, det in enumerate(track.detections):
        cx, cy = det.bottom_center
        if roi.contains_point(cx, cy):
            if run_start_idx is None:
                run_start_idx = i
                run_start_frame = det.frame_idx
        else:
            if run_start_idx is not None:
                _close_run(i - 1, track.detections[i - 1].frame_idx)
                run_start_idx = None
                run_start_frame = None

    if run_start_idx is not None:
        _close_run(len(track.detections) - 1, track.detections[-1].frame_idx)
    return intervals


def _vehicle_detections_in_window(
    track: Track, start_frame: int, end_frame: int
) -> list[Detection]:
    return [d for d in track.detections if start_frame <= d.frame_idx <= end_frame]


def find_violations(
    tracks: list[Track],
    roi: RoiPolygon,
    homography: Homography,
    fps: float,
    config: PipelineConfig,
) -> list[ViolationEvent]:
    """跑完整判定流程。前置:各 track 需已呼叫 compute_track_kinematics。"""
    # ROI 一次性投影到世界座標
    world_roi_pts = homography.image_to_world_many(
        np.asarray(roi.points, dtype=np.float64)
    )
    world_roi = Polygon(world_roi_pts)
    if not world_roi.is_valid:
        raise ValueError("ROI 在世界平面投影後不合法;請檢查 homography 或 ROI 幾何")

    pedestrian_tracks = [t for t in tracks if t.cls == config.pedestrian_class]
    vehicle_tracks = [t for t in tracks if t.cls in config.vehicle_classes]

    violations: list[ViolationEvent] = []
    for ped in pedestrian_tracks:
        intervals = _find_pedestrian_intervals(ped, roi, config.min_pedestrian_frames)
        for interval in intervals:
            for veh in vehicle_tracks:
                event = _check_pair(
                    interval=interval,
                    vehicle=veh,
                    world_roi=world_roi,
                    fps=fps,
                    config=config,
                )
                if event is not None:
                    violations.append(event)
    return violations


def _check_pair(
    *,
    interval: _PedestrianInterval,
    vehicle: Track,
    world_roi: Polygon,
    fps: float,
    config: PipelineConfig,
) -> ViolationEvent | None:
    """對一組 (pedestrian interval, vehicle track) 判斷是否成違規。"""
    veh_in_window = _vehicle_detections_in_window(
        vehicle, interval.start_frame, interval.end_frame
    )
    if not veh_in_window:
        return None

    # 條件 B:車輛到 ROI 的世界距離
    distances: list[float] = []
    dets_with_dist: list[Detection] = []
    for d in veh_in_window:
        if d.world_pos is None:
            continue
        distances.append(world_roi.distance(Point(*d.world_pos)))
        dets_with_dist.append(d)
    if not distances:
        return None

    min_idx = int(np.argmin(distances))
    min_dist = distances[min_idx]
    if min_dist > config.yield_distance_m:
        return None

    closest = dets_with_dist[min_idx]

    # 條件 C:最接近時刻前 window_before_sec 內的最低速度
    # 使用 vehicle.detections 全體,不受 interval 限制——車輛在靠近前的
    # 減速軌跡可能發生於行人進入 ROI 之前。
    window_start = closest.time_sec - config.window_before_sec
    speeds = [
        d.speed_kmh
        for d in vehicle.detections
        if window_start <= d.time_sec <= closest.time_sec and d.speed_kmh is not None
    ]
    if not speeds:
        return None
    min_speed = min(speeds)
    if min_speed <= config.stop_threshold_kmh:
        # 車輛曾近乎停止,視為已停讓
        return None

    return ViolationEvent(
        pedestrian_track_id=interval.track_id,
        vehicle_track_id=vehicle.track_id,
        start_sec=interval.start_frame / fps if fps > 0 else 0.0,
        end_sec=interval.end_frame / fps if fps > 0 else 0.0,
        min_distance_m=float(min_dist),
        min_vehicle_speed_kmh=float(min_speed),
        closest_frame=closest.frame_idx,
        closest_time_sec=closest.time_sec,
    )
