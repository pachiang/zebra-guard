"""YOLO 偵測 + ByteTrack 追蹤。以 Ultralytics 為後端。

**遷移路徑**:目前用 ultralytics 整合 YOLO + ByteTrack 求快。
未來要換成 ONNX Runtime 直接推論時(減少依賴、加上 DirectML 支援):
  1. 新增 `_ort_backend.py`,提供相同的 `(xyxy, conf, cls, id)` per-frame 輸出
  2. `track_video` 依 config 選擇 backend,對外介面不變
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from zebraguard.ml.exceptions import Cancelled
from zebraguard.ml.types import COCO_PERSON, DEFAULT_VEHICLE_CLASSES, Detection


@dataclass(slots=True, frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")
    try:
        return VideoInfo(
            path=str(video_path),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


ProgressCallback = Callable[[int, int], None]  # (current_frame, total_frames)


def track_video(
    video_path: str | Path,
    *,
    model_name: str = "yolo11n.pt",
    tracker: str = "bytetrack.yaml",
    classes: set[int] | frozenset[int] | None = None,
    conf: float = 0.3,
    vid_stride: int = 1,
    imgsz: int = 640,
    max_frames: int | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[VideoInfo, list[Detection]]:
    """跑偵測 + 追蹤,回傳整支影片的所有 tracked detections(扁平)。

    進度:`on_progress(current_frame_idx, total_frame_count)`,
    大約每秒呼叫一次,結束時額外呼叫一次 (total, total)。

    `cancel_event`:set 時下一次迴圈迭代丟 `Cancelled`。
    `max_frames`:若設,超過此 frame_idx 就停。
    """
    from ultralytics import YOLO  # 延遲匯入:純數學測試不需要此依賴

    info = probe_video(video_path)
    if classes is None:
        classes = frozenset({COCO_PERSON, *DEFAULT_VEHICLE_CLASSES})

    model = YOLO(model_name)
    results_iter: Iterator[Any] = model.track(
        source=str(video_path),
        stream=True,
        persist=False,
        tracker=tracker,
        classes=sorted(classes),
        conf=conf,
        imgsz=imgsz,
        vid_stride=vid_stride,
        verbose=False,
    )

    detections: list[Detection] = []
    progress_interval = max(1, int(round(info.fps)) if info.fps > 0 else 30)

    for i, result in enumerate(results_iter):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        frame_idx = i * vid_stride
        if max_frames is not None and frame_idx >= max_frames:
            break
        time_sec = frame_idx / info.fps if info.fps > 0 else 0.0

        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)

            for bbox, det_conf, cls_id, tid in zip(xyxy, conf_arr, cls_arr, ids, strict=False):
                detections.append(
                    Detection(
                        frame_idx=frame_idx,
                        time_sec=float(time_sec),
                        bbox=(
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ),
                        cls=int(cls_id),
                        conf=float(det_conf),
                        track_id=int(tid),
                    )
                )

        if on_progress is not None and (i % progress_interval == 0):
            on_progress(frame_idx, info.frame_count)

    if on_progress is not None:
        on_progress(info.frame_count, info.frame_count)

    return info, detections
