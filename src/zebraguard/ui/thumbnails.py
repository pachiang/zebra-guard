"""事件縮圖產生 + cache。

路徑:`<project>.zgproj/thumbnails/event_<id>.jpg`
尺寸:160 × 90 (HiDPI 環境下 UI 會再縮到 80 × 45 顯示)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

THUMB_W = 200
THUMB_H = 112


def thumbnail_path(project_path: Path, event_id: int) -> Path:
    return project_path / "thumbnails" / f"event_{event_id:05d}.jpg"


def _cached_thumb_is_valid(path: Path) -> bool:
    """檔案存在、非空,且解析度對應目前的 THUMB_W / THUMB_H。"""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    img = cv2.imread(str(path))
    if img is None:
        return False
    return img.shape[0] == THUMB_H and img.shape[1] == THUMB_W


def ensure_event_thumbnails(
    project_path: Path,
    video_path: Path,
    events: list[dict[str, Any]],
    fps: float,
    *,
    force_ids: set[int] | None = None,
) -> dict[int, Path]:
    """為每個事件產生一張 thumbnail(抽中段幀)。

    · 若 cache 已存在且尺寸正確,且 id 不在 `force_ids` 內 → 沿用
    · 否則(含尺寸過期 / trim 後 mid 改變)→ 重新產

    回傳 {event_id: thumb_path}。
    """
    out_dir = project_path / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    force = force_ids or set()

    missing: list[tuple[int, int]] = []  # (event_id, mid_frame)
    result: dict[int, Path] = {}
    for ev in events:
        ev_id = int(ev["id"])
        dst = thumbnail_path(project_path, ev_id)
        if ev_id not in force and _cached_thumb_is_valid(dst):
            result[ev_id] = dst
            continue
        mid_sec = 0.5 * (ev["start_sec"] + ev["end_sec"])
        mid_frame = int(round(mid_sec * fps))
        missing.append((ev_id, mid_frame))

    if not missing:
        return result

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return result
    try:
        for ev_id, frame_idx in missing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            # 等比縮小到 THUMB_W × THUMB_H,中央裁切補不足的軸
            h, w = frame.shape[:2]
            target_ratio = THUMB_W / THUMB_H
            src_ratio = w / h
            if src_ratio > target_ratio:
                # 原圖較寬 → 以高為準裁寬
                new_h = THUMB_H
                new_w = int(round(w * THUMB_H / h))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                x0 = (new_w - THUMB_W) // 2
                cropped = resized[:, x0:x0 + THUMB_W]
            else:
                new_w = THUMB_W
                new_h = int(round(h * THUMB_W / w))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                y0 = (new_h - THUMB_H) // 2
                cropped = resized[y0:y0 + THUMB_H, :]
            dst = thumbnail_path(project_path, ev_id)
            cv2.imwrite(str(dst), cropped, [cv2.IMWRITE_JPEG_QUALITY, 82])
            result[ev_id] = dst
    finally:
        cap.release()
    return result
