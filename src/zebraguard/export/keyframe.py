"""關鍵幀匯出:違規中段擷取一張 .jpg(原圖 + 含 bbox 標註版)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def export_keyframes(
    video_path: Path,
    event: dict[str, Any],
    out_dir: Path,
    fps: float | None = None,
) -> tuple[Path, Path]:
    """從影片擷取事件中段那一幀,輸出兩份 jpg:原圖 + 簡易標註版。

    目前標註版只疊一個「事件時間戳 + 事件 index」,不疊 bbox(因為事件在 project.db
    裡並未逐幀存 bbox)。之後要做更完整的標註圖,應改為在 pipeline 階段快取 bbox。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片:{video_path}")
    try:
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        mid_sec = 0.5 * (event["start_sec"] + event["end_sec"])
        mid_frame = int(round(mid_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"讀取幀失敗 frame={mid_frame}")
    finally:
        cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "keyframe_raw.jpg"
    ann_path = out_dir / "keyframe_annotated.jpg"

    cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 標註版
    annotated = frame.copy()
    m = int(mid_sec) // 60
    s = mid_sec - 60 * m
    tag = f"Event #{event.get('_display_index', event.get('id', 0))}  t={m:02d}:{s:05.2f}"
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(
        annotated, tag, (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return raw_path, ann_path
