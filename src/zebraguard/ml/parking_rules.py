"""違停檢測 · 停留判定 + 違法區內判定 + 候選事件產生。

Pipeline:
  1. YOLO + ByteTrack 產出 Track list(複用 `ml/detection.py`)
  2. `find_stopped_tracks()` 從每條 Track 找出「連續 N 秒位移 ≤ K px」的時段
  3. `bottom_strip_inside_any_roi()` 判斷 bbox 底邊 strip 是否在任一違法區內
  4. `build_parking_candidates()` 綁上面兩者 → 產生違停候選事件清單

規則依據 `docs/parking_detection_plan.md`:
  · 停留判定:同一 track_id 連續時段內 bottom_center 與**起點**距離 ≤ max_disp_px
    (而非與質心或前一幀);起點一超出範圍就切新段,避免緩慢漂移被誤判為停
  · ROI 判定用 bbox 下方 20% strip 的像素覆蓋率,不是整個 bbox;
    這樣車頂斜向跨入違法區但輪子在外不會誤觸
  · **ROI 語意是違法區**(no-parking zone)— 紅黃線 / 人行道 / 禁停標誌。
    使用者不畫 ROI → 沒候選(預設保守,避免亂噴)
  · 預設 `stopped_threshold_sec = 60`,大多紅燈 / 車陣 < 60s 因此自然被濾掉;
    剩餘邊緣案例由使用者在 Review UI 標 `red_light` 兜底(見 plan § 5.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from zebraguard.ml.types import Track

# COCO class ids 預設:car / bus / truck。機車 / 自行車第一版不收(規則不同)
_DEFAULT_PARKING_VEHICLES: frozenset[int] = frozenset({2, 5, 7})


@dataclass(slots=True, frozen=True)
class ParkingConfig:
    stopped_threshold_sec: float = 60.0
    stopped_max_displacement_px: float = 20.0
    # Bottom strip 與**違法區** union 重疊率 >= 此門檻 → 視為「在違法區內」
    roi_overlap_threshold: float = 0.5
    # Bottom strip 佔 bbox 高度的比例
    strip_frac: float = 0.20
    vehicle_classes: frozenset[int] = field(
        default_factory=lambda: _DEFAULT_PARKING_VEHICLES
    )


@dataclass(slots=True)
class ParkingEvent:
    """違停候選事件(單一車輛、連續停留時段)。"""

    vehicle_track_id: int
    vehicle_class: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    # 用於縮圖抽幀 / Review UI 顯示位置
    representative_frame: int
    representative_bbox: tuple[float, float, float, float]

    def to_json(self) -> dict:
        return {
            "vehicle_track_id": self.vehicle_track_id,
            "vehicle_class": self.vehicle_class,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "representative_frame": self.representative_frame,
            "representative_bbox": list(self.representative_bbox),
        }


# =====================================================================
# Stopped detection
# =====================================================================


@dataclass(slots=True)
class _StoppedSegment:
    """內部用:Track 內部的一個連續停留時段。"""

    track_id: int
    cls: int
    start_idx: int  # index in track.detections
    end_idx: int    # inclusive


def find_stopped_tracks(
    track: Track,
    *,
    min_sec: float,
    max_disp_px: float,
) -> list[_StoppedSegment]:
    """從單一 track 找出「bottom_center 相對於段落起點的位移始終 ≤ max_disp_px
    且持續時間 ≥ min_sec」的連續時段。

    起點超出範圍 → 切新段(從該幀起算起點);演算法 O(N)。

    使用 bottom_center(bbox 底邊中心)作為位置代表,這是物件接觸地面的位置,
    對停著的車最穩定(不像 bbox 中心可能因遮擋變動)。
    """
    dets = track.detections
    segments: list[_StoppedSegment] = []
    if not dets:
        return segments

    seg_start_idx = 0
    ref_x, ref_y = dets[0].bottom_center

    def _emit(end_idx: int) -> None:
        dur = dets[end_idx].time_sec - dets[seg_start_idx].time_sec
        if dur >= min_sec:
            segments.append(
                _StoppedSegment(
                    track_id=track.track_id,
                    cls=track.cls,
                    start_idx=seg_start_idx,
                    end_idx=end_idx,
                )
            )

    for i in range(1, len(dets)):
        cx, cy = dets[i].bottom_center
        if hypot(cx - ref_x, cy - ref_y) > max_disp_px:
            _emit(i - 1)
            seg_start_idx = i
            ref_x, ref_y = cx, cy

    _emit(len(dets) - 1)
    return segments


# =====================================================================
# ROI geometry
# =====================================================================


def _build_roi_union(
    roi_polygons: list[list[list[float]]],
) -> Polygon | None:
    """把 list of polygon 合併成一個 shapely geometry(可能是 Polygon 或 MultiPolygon)。

    空列表 / 每個 polygon 頂點 < 3 → None。自相交 polygon 以 buffer(0) 修。
    """
    parts: list[Polygon] = []
    for poly_pts in roi_polygons or []:
        if len(poly_pts) < 3:
            continue
        poly = Polygon([(float(p[0]), float(p[1])) for p in poly_pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        parts.append(poly)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return unary_union(parts)


def bottom_strip_inside_any_roi(
    bbox: tuple[float, float, float, float],
    roi_polygons: list[list[list[float]]],
    *,
    threshold: float = 0.5,
    strip_frac: float = 0.20,
) -> bool:
    """bbox 下方 `strip_frac` 高度的水平 strip 與**違法區** union 的重疊面積
    佔比 **>=** threshold → 回傳 True(視為「在違法區內」)。

    · 無 ROI 時一律 False(沒畫違法區 → 沒候選,預設保守)
    · bbox 退化(寬或高 ≤ 0)→ False(無法判定)
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False

    strip_top = y2 - (y2 - y1) * strip_frac
    strip = box(float(x1), float(strip_top), float(x2), float(y2))
    strip_area = strip.area
    if strip_area <= 0:
        return False

    roi_union = _build_roi_union(roi_polygons)
    if roi_union is None:
        return False

    try:
        overlap = strip.intersection(roi_union).area
    except Exception:  # noqa: BLE001 — shapely 偶有浮點邊界 glitch
        return False
    return (overlap / strip_area) >= threshold


# =====================================================================
# Candidate generation
# =====================================================================


def build_parking_candidates(
    tracks: list[Track],
    no_parking_zones: list[list[list[float]]],
    *,
    config: ParkingConfig | None = None,
) -> list[ParkingEvent]:
    """跑完整違停候選判定流程。"""
    cfg = config or ParkingConfig()
    candidates: list[ParkingEvent] = []

    for track in tracks:
        if track.cls not in cfg.vehicle_classes:
            continue
        segs = find_stopped_tracks(
            track,
            min_sec=cfg.stopped_threshold_sec,
            max_disp_px=cfg.stopped_max_displacement_px,
        )
        for seg in segs:
            # 取代表幀為時段中點,bbox 是該幀的 bbox
            mid_idx = (seg.start_idx + seg.end_idx) // 2
            rep = track.detections[mid_idx]
            if not bottom_strip_inside_any_roi(
                rep.bbox,
                no_parking_zones,
                threshold=cfg.roi_overlap_threshold,
                strip_frac=cfg.strip_frac,
            ):
                continue
            start_det = track.detections[seg.start_idx]
            end_det = track.detections[seg.end_idx]
            candidates.append(
                ParkingEvent(
                    vehicle_track_id=track.track_id,
                    vehicle_class=track.cls,
                    start_frame=start_det.frame_idx,
                    end_frame=end_det.frame_idx,
                    start_sec=start_det.time_sec,
                    end_sec=end_det.time_sec,
                    representative_frame=rep.frame_idx,
                    representative_bbox=rep.bbox,
                )
            )
    return candidates
