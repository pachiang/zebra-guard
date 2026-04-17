"""把扁平 Detection 列表聚合成 Track,並計算世界位置與瞬時速度。

**已知限制(MVP)**:速度直接由相鄰幀位置差計算,沒有平滑。
真實影片因 bbox 抖動,速度訊號會含雜訊。Phase 2 應加入 Kalman
或 EMA 平滑(見 docs/mvp.md)。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from zebraguard.ml.homography import Homography
from zebraguard.ml.types import Detection, Track


def aggregate_tracks(detections: Iterable[Detection]) -> list[Track]:
    """依 track_id 分組;無 track_id 的偵測忽略。每個 Track 依 frame_idx 升冪排序。"""
    buckets: dict[int, list[Detection]] = defaultdict(list)
    classes: dict[int, int] = {}
    for det in detections:
        if det.track_id is None:
            continue
        buckets[det.track_id].append(det)
        classes[det.track_id] = det.cls

    tracks: list[Track] = []
    for tid, dets in buckets.items():
        dets.sort(key=lambda d: d.frame_idx)
        tracks.append(Track(track_id=tid, cls=classes[tid], detections=dets))
    return tracks


def compute_track_kinematics(track: Track, homography: Homography) -> None:
    """In-place 填入每個 Detection 的 `world_pos` 與 `speed_kmh`。

    速度 = ‖Δ世界位置‖ / Δt,單位 km/h。第一個偵測 speed_kmh = 0。
    """
    if not track.detections:
        return

    bottom_centers = np.asarray([d.bottom_center for d in track.detections], dtype=np.float64)
    world_positions = homography.image_to_world_many(bottom_centers)

    prev_pos: tuple[float, float] | None = None
    prev_time: float | None = None
    for det, (wx, wy) in zip(track.detections, world_positions, strict=True):
        det.world_pos = (float(wx), float(wy))
        if prev_pos is None or prev_time is None:
            # 無前一幀可比對:無速度資料(None 有別於真實的 0)
            det.speed_kmh = None
        else:
            dt = det.time_sec - prev_time
            if dt <= 0:
                det.speed_kmh = None
            else:
                dx = wx - prev_pos[0]
                dy = wy - prev_pos[1]
                speed_m_per_s = float(np.hypot(dx, dy)) / dt
                det.speed_kmh = speed_m_per_s * 3.6
        prev_pos = (float(wx), float(wy))
        prev_time = det.time_sec
