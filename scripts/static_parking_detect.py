"""違停檢測 pipeline(靜態攝影機)— YOLO + ByteTrack + parking_rules。

與 `zero_shot_detect.py` 並列;不使用 Mask2Former / YOLO-seg / crosswalk source,
也不使用 homography。核心流程:

  1. YOLO + ByteTrack 追人 / 車(`ml/detection.py:track_video`)
  2. Track 聚合(`ml/tracking.py:aggregate_tracks`)
  3. `ml/parking_rules.py:build_parking_candidates`(停留判定 + ROI 外判定)
  4. 回傳 report dict(candidates list)

對外 API 與 `zero_shot_detect.run()` 一致:progress_cb(stage, current, total, hits)
+ cancel_event + 回傳 report。worker 依 project.meta.mode 選擇呼叫哪個。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

# 讓 script 直接執行時也能 import zebraguard
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from zebraguard.ml.detection import probe_video, track_video
from zebraguard.ml.exceptions import Cancelled  # noqa: F401 — 對外 re-export
from zebraguard.ml.parking_rules import (
    ParkingConfig,
    build_parking_candidates,
)
from zebraguard.ml.tracking import aggregate_tracks
from zebraguard.ml.types import COCO_BUS, COCO_CAR, COCO_TRUCK

ProgressCallback = Callable[[str, int, int, int], None]


def _log(msg: str) -> None:
    print(msg, flush=True)


def run(
    video_path: Path,
    *,
    no_parking_zones: list[list[list[float]]],
    yolo_weights: str = "yolo11n.pt",
    conf: float = 0.3,
    imgsz: int = 640,
    vid_stride: int = 1,
    stopped_threshold_sec: float = 60.0,
    stopped_max_displacement_px: float = 20.0,
    roi_overlap_threshold: float = 0.5,
    strip_frac: float = 0.20,
    vehicle_classes: list[int] | None = None,
    max_seconds: float | None = None,
    output_json: Path | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    info = probe_video(video_path)
    _log(
        f"[video] {video_path.name}: {info.width}x{info.height} @ "
        f"{info.fps:.2f}fps  {info.frame_count} frames "
        f"({info.duration_sec:.1f}s)"
    )
    max_frame = info.frame_count
    if max_seconds is not None and info.fps > 0:
        max_frame = min(info.frame_count, int(max_seconds * info.fps))
        _log(f"[video] --max-seconds={max_seconds} → 處理首 {max_frame} 幀")

    classes_set = frozenset(vehicle_classes or [COCO_CAR, COCO_BUS, COCO_TRUCK])

    _log(f"[models] 載入 YOLO {yolo_weights}…")
    if progress_cb is not None:
        progress_cb("loading_yolo", 0, 1, 0)

    def _on_detect_progress(current: int, total: int) -> None:
        if progress_cb is not None:
            progress_cb("analyzing", current, max_frame or total, 0)

    _log("[track] YOLO + ByteTrack…")
    t0 = time.monotonic()
    _, detections = track_video(
        video_path,
        model_name=yolo_weights,
        classes=classes_set,
        conf=conf,
        imgsz=imgsz,
        vid_stride=vid_stride,
        max_frames=max_frame,
        on_progress=_on_detect_progress,
        cancel_event=cancel_event,
    )
    _log(f"[track] 完成 {len(detections)} dets in {time.monotonic() - t0:.1f}s")

    if progress_cb is not None:
        progress_cb("analyzing", max_frame, max_frame, 0)

    tracks = aggregate_tracks(detections)
    _log(f"[track] 聚合 → {len(tracks)} tracks")

    cfg = ParkingConfig(
        stopped_threshold_sec=stopped_threshold_sec,
        stopped_max_displacement_px=stopped_max_displacement_px,
        roi_overlap_threshold=roi_overlap_threshold,
        strip_frac=strip_frac,
        vehicle_classes=classes_set,
    )
    candidates = build_parking_candidates(tracks, no_parking_zones, config=cfg)
    _log(f"[rules] 產出 {len(candidates)} 筆違停候選")

    for i, c in enumerate(candidates):
        _log(
            f"  #{i + 1}: track {c.vehicle_track_id}  "
            f"cls={c.vehicle_class}  "
            f"t=[{c.start_sec:.1f}–{c.end_sec:.1f}s] "
            f"({c.end_sec - c.start_sec:.1f}s)"
        )

    if progress_cb is not None:
        progress_cb("done", max_frame, max_frame, len(candidates))

    report = {
        "video": str(video_path),
        "fps": info.fps,
        "frame_count": max_frame,
        "duration_sec": max_frame / info.fps if info.fps > 0 else 0.0,
        "candidates": [c.to_json() for c in candidates],
        "params": {
            "mode": "static",
            "yolo_weights": yolo_weights,
            "conf": conf,
            "imgsz": imgsz,
            "vid_stride": vid_stride,
            "stopped_threshold_sec": stopped_threshold_sec,
            "stopped_max_displacement_px": stopped_max_displacement_px,
            "roi_overlap_threshold": roi_overlap_threshold,
            "strip_frac": strip_frac,
            "vehicle_classes": sorted(classes_set),
            "max_seconds": max_seconds,
            "n_no_parking_zones": len(no_parking_zones or []),
        },
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log(f"[done] report: {output_json}")
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--no-parking-zones-json", type=Path, required=False,
                   help="JSON 含 list of polygon(違法區,如紅黃線/人行道);"
                        "每個 polygon 是 list of [x,y]。省略 → 沒候選(保守)")
    p.add_argument("--out", type=Path, default=Path("static_parking_events.json"))
    p.add_argument("--yolo", type=str, default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--stopped-sec", type=float, default=60.0)
    p.add_argument("--stopped-disp-px", type=float, default=20.0)
    p.add_argument("--roi-overlap-thresh", type=float, default=0.5)
    p.add_argument("--strip-frac", type=float, default=0.20)
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args()

    zones: list[list[list[float]]] = []
    if args.no_parking_zones_json and args.no_parking_zones_json.is_file():
        with open(args.no_parking_zones_json, encoding="utf-8") as f:
            zones = json.load(f)

    run(
        args.video,
        no_parking_zones=zones,
        yolo_weights=args.yolo,
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=args.stride,
        stopped_threshold_sec=args.stopped_sec,
        stopped_max_displacement_px=args.stopped_disp_px,
        roi_overlap_threshold=args.roi_overlap_thresh,
        strip_frac=args.strip_frac,
        max_seconds=args.max_seconds,
        output_json=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
