"""Pipeline 編排:影片 → 偵測追蹤 → kinematics → 違規判定。

本模組**不依賴 PySide6**。UI 透過 `on_progress` callback 串進度;
未來在 core/worker.py 以 QThread 包裝本函式並把 callback 轉成 Qt Signal。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zebraguard.ml.detection import ProgressCallback, VideoInfo, track_video
from zebraguard.ml.homography import Homography
from zebraguard.ml.roi import RoiPolygon
from zebraguard.ml.tracking import aggregate_tracks, compute_track_kinematics
from zebraguard.ml.types import PipelineConfig, ViolationEvent
from zebraguard.ml.violation_rules import find_violations


@dataclass(slots=True)
class PipelineResult:
    video: VideoInfo
    violations: list[ViolationEvent]
    n_tracks: int


def run_pipeline(
    *,
    video_path: str | Path,
    roi: RoiPolygon,
    homography: Homography,
    config: PipelineConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> PipelineResult:
    """跑完整管線。

    目前為單階段(Stage 1+2 合一)。MVP P1 再拆分以省 12 小時影片的成本:
    粗篩低 fps → 精細於可疑時段高 fps。
    """
    cfg = config or PipelineConfig()

    info, detections = track_video(
        video_path,
        model_name=cfg.model_name,
        tracker=cfg.tracker,
        classes=frozenset({cfg.pedestrian_class, *cfg.vehicle_classes}),
        conf=cfg.detection_conf,
        vid_stride=cfg.vid_stride,
        on_progress=on_progress,
    )

    tracks = aggregate_tracks(detections)
    for t in tracks:
        compute_track_kinematics(t, homography)

    violations = find_violations(
        tracks=tracks,
        roi=roi,
        homography=homography,
        fps=info.fps,
        config=cfg,
    )
    return PipelineResult(video=info, violations=violations, n_tracks=len(tracks))
