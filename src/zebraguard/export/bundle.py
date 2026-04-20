"""把所有採用事件匯成一個資料夾結構。

每個事件一個子資料夾:
  out_root/
    event_001_mm-ss/
      clip.mp4
      keyframe_raw.jpg
      keyframe_annotated.jpg
      complaint_draft.txt
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zebraguard import __version__
from zebraguard.export.clip import extract_clip
from zebraguard.export.complaint import write_complaint_draft
from zebraguard.export.keyframe import export_keyframes

ExportProgressCb = Callable[[int, int, str], None]  # (current, total, message)

EXPORT_LAYOUT = """\
event_NNN_MM-SS/
  clip.mp4                — 違規片段(違規前後各 5 秒)
  keyframe_raw.jpg        — 中段原圖
  keyframe_annotated.jpg  — 中段含時間戳標註
  complaint_draft.txt     — 檢舉書草稿
"""


def _event_folder_name(event: dict[str, Any]) -> str:
    idx = event.get("_display_index", event.get("id", 0))
    sec = event["start_sec"]
    m = int(sec) // 60
    s = int(sec) % 60
    return f"event_{idx:03d}_{m:02d}-{s:02d}"


def export_accepted_events(
    video_path: Path,
    events: list[dict[str, Any]],
    out_root: Path,
    *,
    fps: float | None = None,
    progress_cb: ExportProgressCb | None = None,
) -> list[Path]:
    """依序匯出每個事件。回傳各事件的輸出資料夾 list。"""
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    total = len(events)
    for i, ev in enumerate(events):
        folder = out_root / _event_folder_name(ev)
        folder.mkdir(parents=True, exist_ok=True)

        if progress_cb is not None:
            progress_cb(i, total, f"匯出 {folder.name} / 片段…")
        extract_clip(video_path, ev["start_sec"], ev["end_sec"], folder / "clip.mp4")

        if progress_cb is not None:
            progress_cb(i, total, f"匯出 {folder.name} / 關鍵幀…")
        export_keyframes(video_path, ev, folder, fps=fps)

        if progress_cb is not None:
            progress_cb(i, total, f"匯出 {folder.name} / 檢舉書草稿…")
        write_complaint_draft(
            folder / "complaint_draft.txt",
            event=ev,
            video_name=video_path.name,
            total_events=total,
            app_version=__version__,
        )
        results.append(folder)

    if progress_cb is not None:
        progress_cb(total, total, "匯出完成")
    return results
