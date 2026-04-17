"""共用資料結構。純 dataclass,無任何 ML 框架相依。"""

from __future__ import annotations

from dataclasses import dataclass, field

# COCO 類別 ID
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7

DEFAULT_VEHICLE_CLASSES: frozenset[int] = frozenset(
    {COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}
)


@dataclass(slots=True)
class Detection:
    """單幀、單個物件的偵測結果。"""

    frame_idx: int
    time_sec: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) 影像像素座標
    cls: int
    conf: float
    track_id: int | None = None

    # 後續 kinematics 計算填入
    world_pos: tuple[float, float] | None = None  # 底邊中心的世界座標 (m)
    speed_kmh: float | None = None  # 瞬時速度

    @property
    def bottom_center(self) -> tuple[float, float]:
        """bbox 底邊中心點(假設為物件與地面接觸點)。"""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass(slots=True)
class Track:
    """同一個 track_id 的所有偵測,依 frame_idx 升冪排序。"""

    track_id: int
    cls: int
    detections: list[Detection]

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx


@dataclass(slots=True)
class ViolationEvent:
    """一筆未禮讓違規候選事件。使用者尚需在 UI 審查後才視為確認。"""

    pedestrian_track_id: int
    vehicle_track_id: int
    start_sec: float
    end_sec: float
    min_distance_m: float
    min_vehicle_speed_kmh: float
    closest_frame: int
    closest_time_sec: float

    def to_json(self) -> dict:
        return {
            "pedestrian_track_id": self.pedestrian_track_id,
            "vehicle_track_id": self.vehicle_track_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "min_distance_m": self.min_distance_m,
            "min_vehicle_speed_kmh": self.min_vehicle_speed_kmh,
            "closest_frame": self.closest_frame,
            "closest_time_sec": self.closest_time_sec,
        }


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """管線參數。對應 docs/legal-rules.md § 3 的三條件。"""

    # 規則判定
    yield_distance_m: float = 3.0
    stop_threshold_kmh: float = 5.0
    window_before_sec: float = 2.0
    min_pedestrian_frames: int = 3

    # 類別過濾
    pedestrian_class: int = COCO_PERSON
    vehicle_classes: frozenset[int] = field(default_factory=lambda: DEFAULT_VEHICLE_CLASSES)

    # 偵測
    detection_conf: float = 0.3
    vid_stride: int = 1
    model_name: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
