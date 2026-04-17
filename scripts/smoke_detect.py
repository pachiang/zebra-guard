"""Detection-only smoke test:跑 YOLO + ByteTrack,輸出偵測統計 + 標註影片。

不做 ROI / homography / 違規判定(那些需要靜態攝影機與校正)。
目的只是驗證:模型能載入、影片能解碼、追蹤能跑出連續 track_id。

Usage:
    python scripts/smoke_detect.py --video path/to/video.mp4 \\
        [--model resources/models/yolo11n.pt] \\
        [--out-video annotated.mp4]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# line-buffer stdout 才能在背景執行時即時看到進度
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

import cv2  # noqa: E402

from zebraguard.ml.types import (  # noqa: E402
    COCO_BICYCLE,
    COCO_BUS,
    COCO_CAR,
    COCO_MOTORCYCLE,
    COCO_PERSON,
    COCO_TRUCK,
)

CLASS_NAMES = {
    COCO_PERSON: "person",
    COCO_BICYCLE: "bicycle",
    COCO_CAR: "car",
    COCO_MOTORCYCLE: "motorcycle",
    COCO_BUS: "bus",
    COCO_TRUCK: "truck",
}

CLASS_COLORS = {
    COCO_PERSON: (0, 255, 0),
    COCO_BICYCLE: (255, 165, 0),
    COCO_CAR: (0, 0, 255),
    COCO_MOTORCYCLE: (255, 0, 255),
    COCO_BUS: (255, 255, 0),
    COCO_TRUCK: (128, 0, 255),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "resources" / "models" / "yolo11n.pt"
    )
    parser.add_argument("--out-video", type=Path, default=Path("annotated.mp4"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    _log(f"[1/4] 載入模型 {args.model}")
    t0 = time.monotonic()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    _log(f"      完成 ({time.monotonic() - t0:.1f}s)")

    _log(f"[2/4] 探測影片 {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        _log(f"無法開啟 {args.video}")
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    _log(f"      {w}x{h}, {fps:.2f} fps, {total} frames")

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out_video), fourcc, fps, (w, h))

    cls_counts: Counter[int] = Counter()
    tracks_seen: set[int] = set()
    tracks_by_cls: dict[int, set[int]] = defaultdict(set)
    frames_with_detection = 0

    interesting = {COCO_PERSON, COCO_BICYCLE, COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}

    _log(f"[3/4] 追蹤中...")
    t_start = time.monotonic()
    last_log = 0.0

    results_iter = model.track(
        source=str(args.video),
        stream=True,
        persist=False,
        tracker="bytetrack.yaml",
        classes=sorted(interesting),
        conf=args.conf,
        verbose=False,
    )

    for i, result in enumerate(results_iter):
        frame = result.orig_img.copy() if result.orig_img is not None else None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            frames_with_detection += 1
            xyxy = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for bbox, cls_id, tid, c in zip(xyxy, cls_arr, ids, confs, strict=False):
                cls_counts[int(cls_id)] += 1
                tracks_seen.add(int(tid))
                tracks_by_cls[int(cls_id)].add(int(tid))
                if frame is not None:
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
                    label = (
                        f"{CLASS_NAMES.get(int(cls_id), str(int(cls_id)))}"
                        f"#{int(tid)} {c:.2f}"
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        if frame is not None:
            writer.write(frame)

        # 每秒一行進度
        now = time.monotonic()
        if now - last_log >= 1.0 or i == total - 1:
            last_log = now
            elapsed = now - t_start
            cur = i + 1
            proc_fps = cur / elapsed if elapsed > 0 else 0.0
            eta = (total - cur) / proc_fps if proc_fps > 0 else 0.0
            _log(
                f"      {cur}/{total} ({100.0 * cur / total:5.1f}%)  "
                f"{proc_fps:5.1f} fps  ETA {eta:5.1f}s  "
                f"tracks seen: {len(tracks_seen)}"
            )

    writer.release()
    elapsed = time.monotonic() - t_start
    proc_fps = total / elapsed if elapsed > 0 else 0.0

    _log("")
    _log("[4/4] 結果")
    _log("=" * 60)
    _log(f"耗時:        {elapsed:.2f}s")
    _log(f"處理速度:    {proc_fps:.1f} fps")
    _log(f"有偵測的幀:  {frames_with_detection} / {total}")
    _log(f"獨立 tracks: {len(tracks_seen)}")
    _log("各類別:")
    for cls_id, n in cls_counts.most_common():
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        n_tracks = len(tracks_by_cls[cls_id])
        _log(f"  {name:12s}  {n:5d} detections / {n_tracks} tracks")
    _log(f"標註影片:    {args.out_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
