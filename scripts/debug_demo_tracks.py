"""Debug:為何 demo 跑出 0 違規?把 tracks 細節印出來。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

import numpy as np  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402

from zebraguard.ml.detection import track_video  # noqa: E402
from zebraguard.ml.homography import Homography  # noqa: E402
from zebraguard.ml.roi import RoiPolygon  # noqa: E402
from zebraguard.ml.tracking import aggregate_tracks, compute_track_kinematics  # noqa: E402
from zebraguard.ml.types import COCO_PERSON, DEFAULT_VEHICLE_CLASSES  # noqa: E402

CLIP = ROOT / "tests" / "fixtures" / "roadsafety_clip.mp4"
ROI_JSON = ROOT / "tests" / "fixtures" / "roadsafety_roi.json"
HOM_JSON = ROOT / "tests" / "fixtures" / "roadsafety_hom.json"

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    roi = RoiPolygon.load(ROI_JSON)
    hom = Homography.load(HOM_JSON)

    _log("跑追蹤...")
    info, dets = track_video(
        CLIP,
        model_name=str(ROOT / "resources" / "models" / "yolo11n.pt"),
        classes=frozenset({COCO_PERSON, *DEFAULT_VEHICLE_CLASSES}),
    )
    _log(f"共 {len(dets)} detections  影片 {info.fps}fps {info.frame_count}幀")

    tracks = aggregate_tracks(dets)
    for t in tracks:
        compute_track_kinematics(t, hom)

    # 世界座標 ROI
    world_poly_pts = hom.image_to_world_many(np.asarray(roi.points, dtype=np.float64))
    world_roi = Polygon(world_poly_pts)
    _log(f"\nROI(世界座標):")
    for i, (wx, wy) in enumerate(world_poly_pts):
        ix, iy = roi.points[i]
        _log(f"  P{i + 1}: image ({ix}, {iy}) → world ({wx:.2f}, {wy:.2f})")
    _log(f"  ROI area: {world_roi.area:.2f} m^2")

    _log(f"\n{len(tracks)} tracks:")
    _log(f"{'id':>4} {'cls':<11} {'frames':>7} {'first→last':>13} "
         f"{'bbox底中心(影→世)':<30} {'speed km/h':<10} {'minDist→ROI':<10} {'inROI':<6}")

    for t in sorted(tracks, key=lambda x: x.track_id):
        n = len(t.detections)
        first = t.detections[0]
        last = t.detections[-1]
        fx, fy = first.bottom_center
        first_world = first.world_pos or (0.0, 0.0)
        speeds = [d.speed_kmh for d in t.detections if d.speed_kmh is not None]
        max_speed = max(speeds) if speeds else 0.0

        # 該 track 到 world ROI 的最短距離(取所有偵測點的最小)
        min_dist = float("inf")
        any_in_roi = False
        for d in t.detections:
            if d.world_pos is None:
                continue
            p = Point(*d.world_pos)
            dist = world_roi.distance(p)
            if dist < min_dist:
                min_dist = dist
            # ROI 判定使用影像座標(行人bottom_center)
            if roi.contains_point(*d.bottom_center):
                any_in_roi = True

        cls_name = CLASS_NAMES.get(t.cls, str(t.cls))
        _log(
            f"{t.track_id:>4} {cls_name:<11} {n:>7} "
            f"{first.frame_idx:>3}→{last.frame_idx:<9} "
            f"({fx:>6.0f},{fy:>4.0f}) ({first_world[0]:>6.1f},{first_world[1]:>5.1f}) "
            f"max {max_speed:>5.1f}  "
            f"{min_dist:>8.2f}m   "
            f"{'YES' if any_in_roi else '---'}"
        )

    _log("\n類別總數:")
    cls_counts = Counter(t.cls for t in tracks)
    for cls_id, n in cls_counts.items():
        _log(f"  {CLASS_NAMES.get(cls_id, str(cls_id)):10s} {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
