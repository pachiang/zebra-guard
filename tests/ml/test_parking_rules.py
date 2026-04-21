"""違停判定規則 parking_rules 的單元測試。"""

from __future__ import annotations

from zebraguard.ml.parking_rules import (
    ParkingConfig,
    bottom_strip_inside_any_roi,
    build_parking_candidates,
    find_stopped_tracks,
)
from zebraguard.ml.types import COCO_CAR, COCO_PERSON, Detection, Track


# =========================================================================
# Helpers
# =========================================================================


def _det(frame: int, fps: float, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(
        frame_idx=frame,
        time_sec=frame / fps,
        bbox=bbox,
        cls=COCO_CAR,
        conf=0.9,
        track_id=1,
    )


def _track(dets: list[Detection], cls: int = COCO_CAR, track_id: int = 1) -> Track:
    for d in dets:
        d.track_id = track_id
        d.cls = cls
    return Track(track_id=track_id, cls=cls, detections=dets)


def _static_bbox(x: float, y: float, w: float = 100, h: float = 60):
    return (x, y, x + w, y + h)


# =========================================================================
# find_stopped_tracks
# =========================================================================


def test_empty_track_returns_no_segments():
    track = _track([])
    assert find_stopped_tracks(track, min_sec=60.0, max_disp_px=20.0) == []


def test_fully_static_long_track_one_segment():
    # 10 秒靜止 + 閾值 5 秒 → 1 segment
    fps = 30.0
    dets = [_det(i, fps, _static_bbox(500, 400)) for i in range(0, 300, 30)]
    track = _track(dets)
    segs = find_stopped_tracks(track, min_sec=5.0, max_disp_px=5.0)
    assert len(segs) == 1
    assert segs[0].start_idx == 0
    assert segs[0].end_idx == len(dets) - 1


def test_fully_static_but_too_short_returns_none():
    # 3 秒靜止,閾值 5 秒 → 無 segment
    fps = 30.0
    dets = [_det(i, fps, _static_bbox(500, 400)) for i in range(0, 90, 30)]
    track = _track(dets)
    segs = find_stopped_tracks(track, min_sec=5.0, max_disp_px=5.0)
    assert segs == []


def test_moving_track_not_stopped():
    # bottom_center 每幀位移 10 px,max_disp_px=5 → 每 step 都打破 → 無 segment
    fps = 30.0
    dets = [_det(i, fps, _static_bbox(500 + i * 10, 400)) for i in range(0, 10)]
    track = _track(dets)
    segs = find_stopped_tracks(track, min_sec=0.1, max_disp_px=5.0)
    # 每個 single-point segment 的 duration = 0,不符 min_sec
    assert segs == []


def test_stop_then_move_then_stop_gives_two_segments():
    # 0-3s 靜止於 (500, 400);3-4s 移到 (700, 400);4-8s 靜止於 (700, 400)
    fps = 30.0
    dets: list[Detection] = []
    for f in range(0, 90, 30):  # 0, 1, 2 秒
        dets.append(_det(f, fps, _static_bbox(500, 400)))
    # 移動幀
    dets.append(_det(120, fps, _static_bbox(700, 400)))
    for f in range(150, 240, 30):  # 5, 6, 7 秒
        dets.append(_det(f, fps, _static_bbox(700, 400)))
    track = _track(dets)
    segs = find_stopped_tracks(track, min_sec=2.0, max_disp_px=5.0)
    assert len(segs) == 2
    # 第一段結束 idx 應在移動點之前
    assert segs[0].start_idx == 0
    assert dets[segs[0].end_idx].bbox[0] == 500
    # 第二段起始就是移動後的 (700, 400) 那個幀
    assert dets[segs[1].start_idx].bbox[0] == 700


def test_slow_drift_exceeds_max_disp_splits_segment():
    # 每幀 bottom_center 緩慢位移 3 px,10 幀累積超過 max_disp=10
    fps = 30.0
    dets = [_det(i * 30, fps, _static_bbox(500 + i * 3, 400)) for i in range(0, 10)]
    track = _track(dets)
    segs = find_stopped_tracks(track, min_sec=2.0, max_disp_px=10.0)
    # 前 4 幀(0-90 frame, 0-3 秒)偏移 0-9 px <= 10 → 同段;
    # 第 5 幀(120 frame)偏移 12 px → 破段
    # 但 0-3s 只有 3 秒 >= 2s → 第一段成立
    assert len(segs) >= 1


# =========================================================================
# bottom_strip_inside_any_roi(違法區語意)
# =========================================================================


def test_empty_roi_never_triggers():
    # 沒畫違法區 → 所有 bbox 都不觸發
    assert bottom_strip_inside_any_roi((0, 0, 100, 100), []) is False
    assert bottom_strip_inside_any_roi((0, 0, 100, 100), [[]]) is False
    assert bottom_strip_inside_any_roi((0, 0, 100, 100), [[[0, 0], [10, 10]]]) is False


def test_bbox_fully_inside_roi_triggers():
    no_park = [[[0, 0], [500, 0], [500, 500], [0, 500]]]
    assert bottom_strip_inside_any_roi((100, 100, 200, 200), no_park) is True


def test_bbox_fully_outside_roi_no_trigger():
    no_park = [[[0, 0], [100, 0], [100, 100], [0, 100]]]
    assert bottom_strip_inside_any_roi((500, 500, 600, 600), no_park) is False


def test_top_inside_bottom_outside_no_trigger():
    """車頂斜向跨入違法區但輪子在外 → 不觸發(物理位置不在違法區)。"""
    no_park = [[[0, 0], [1000, 0], [1000, 200], [0, 200]]]  # 畫面上半
    # bbox y=150..400:bbox 頂部進違法區,但 bottom strip (y=350..400) 在外
    assert bottom_strip_inside_any_roi((300, 150, 400, 400), no_park) is False


def test_partial_overlap_threshold_boundary():
    # 違法區右半邊;bbox bottom strip 50% 在內 → 恰達 0.5 門檻 → 觸發
    no_park = [[[500, 0], [1000, 0], [1000, 1000], [500, 1000]]]
    assert bottom_strip_inside_any_roi((400, 100, 600, 200), no_park, threshold=0.5) is True
    # 門檻拉到 0.7 → 0.5 < 0.7 → 不觸發
    assert bottom_strip_inside_any_roi((400, 100, 600, 200), no_park, threshold=0.7) is False


def test_degenerate_bbox_no_trigger():
    assert bottom_strip_inside_any_roi((100, 100, 100, 200), [[[0, 0], [500, 0], [500, 500], [0, 500]]]) is False
    assert bottom_strip_inside_any_roi((100, 200, 200, 200), [[[0, 0], [500, 0], [500, 500], [0, 500]]]) is False


def test_multiple_rois_union_correct():
    no_park = [
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[500, 500], [700, 500], [700, 700], [500, 700]],
    ]
    # bbox 在第二個違法區內 → 觸發
    assert bottom_strip_inside_any_roi((520, 520, 680, 680), no_park) is True
    # bbox 兩區外 → 不觸發
    assert bottom_strip_inside_any_roi((300, 300, 400, 400), no_park) is False


def test_self_intersecting_polygon_handled():
    butterfly = [[[0, 0], [100, 100], [100, 0], [0, 100]]]
    result = bottom_strip_inside_any_roi((30, 30, 50, 50), butterfly)
    assert isinstance(result, bool)


# =========================================================================
# build_parking_candidates 整合
# =========================================================================


def test_build_candidates_end_to_end():
    """違法區蓋在 (500, 400) 那一帶;只有停在違法區內的車要出現。"""
    fps = 30.0
    # Track A: 停在 (500, 400) 共 70 秒 + 在違法區內 → 候選
    dets_a = [_det(f, fps, _static_bbox(500, 400)) for f in range(0, 2100 + 1, 60)]
    track_a = _track(dets_a, cls=COCO_CAR, track_id=1)

    # Track B: 停在違法區外 → 不出現
    dets_b = [_det(f, fps, _static_bbox(100, 100)) for f in range(0, 2100 + 1, 60)]
    track_b = _track(dets_b, cls=COCO_CAR, track_id=2)

    # Track C: 是行人 → 類別不收
    dets_c = [_det(f, fps, _static_bbox(500, 400)) for f in range(0, 2100 + 1, 60)]
    track_c = _track(dets_c, cls=COCO_PERSON, track_id=3)

    # 違法區蓋住 (500, 400) bottom strip 的位置
    # bbox 500..600, 400..460;bottom strip y=448..460;違法區要覆蓋此 strip
    no_parking = [[[400, 440], [700, 440], [700, 500], [400, 500]]]

    cfg = ParkingConfig(stopped_threshold_sec=60.0, stopped_max_displacement_px=10.0)
    candidates = build_parking_candidates(
        [track_a, track_b, track_c], no_parking, config=cfg
    )
    assert len(candidates) == 1
    assert candidates[0].vehicle_track_id == 1
    assert candidates[0].vehicle_class == COCO_CAR
    assert candidates[0].start_sec == 0.0
    assert candidates[0].end_sec >= 60.0


def test_build_candidates_moving_car_excluded():
    fps = 30.0
    # Track 移動中 → 無停留段;即使在違法區內也不觸發
    dets = [_det(f, fps, _static_bbox(100 + f, 400)) for f in range(0, 3000 + 1, 30)]
    track = _track(dets, cls=COCO_CAR, track_id=1)
    no_parking = [[[0, 0], [10000, 0], [10000, 1000], [0, 1000]]]
    candidates = build_parking_candidates(
        [track], no_parking,
        config=ParkingConfig(stopped_threshold_sec=60.0, stopped_max_displacement_px=10.0),
    )
    assert candidates == []


def test_build_candidates_no_roi_no_candidates():
    """沒畫違法區 → 即使停再久也不產生候選(預設保守)。"""
    fps = 30.0
    dets = [_det(f, fps, _static_bbox(150, 150)) for f in range(0, 2100 + 1, 30)]
    track = _track(dets, cls=COCO_CAR, track_id=1)
    candidates = build_parking_candidates([track], [])
    assert candidates == []


def test_build_candidates_stopped_outside_roi_excluded():
    """停 70 秒但不在違法區 → 不觸發。"""
    fps = 30.0
    dets = [_det(f, fps, _static_bbox(150, 150)) for f in range(0, 2100 + 1, 30)]
    track = _track(dets, cls=COCO_CAR, track_id=1)
    # 違法區在遠處
    no_parking = [[[800, 800], [900, 800], [900, 900], [800, 900]]]
    candidates = build_parking_candidates([track], no_parking)
    assert candidates == []
