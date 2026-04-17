"""CLI 入口。`uv run zebraguard-pipeline ...` 或 `python scripts/run_pipeline.py ...`。

用法(兩選一 — 輸出格式):
  A. 寫扁平 JSON(除錯用):
     zebraguard-pipeline --video V.mp4 --roi r.json --homography h.json --output out.json

  B. 寫進 Project (.zgproj)(建議):
     zebraguard-pipeline --video V.mp4 --roi r.json --homography h.json --project my.zgproj

roi.json:
    {"polygon": [[x, y], [x, y], [x, y], [x, y]]}

homography.json:
    {"image_points": [[x, y], ...], "world_points_meters": [[x, y], ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from zebraguard.core.pipeline import PipelineResult, run_pipeline
from zebraguard.core.project import Project
from zebraguard.ml.homography import Homography
from zebraguard.ml.roi import RoiPolygon
from zebraguard.ml.types import PipelineConfig


def _enable_line_buffering() -> None:
    """把 stdout/stderr 改成 line-buffered。

    Windows 下 Python stdout 重導向到檔案或 pipe 時預設 block-buffered,
    需要手動切到 line-buffered 才能即時看到輸出。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except AttributeError:
            pass


def _build_progress_printer():
    """進度回報。

    TTY 下用 \\r 覆寫單一行;非 TTY 下用換行式 log,每行都會被 line-buffer
    立刻 flush 出來。
    """
    is_tty = sys.stdout.isatty()
    last_time = {"t": 0.0}
    start = time.monotonic()

    def cb(current: int, total: int) -> None:
        now = time.monotonic()
        if now - last_time["t"] < 1.0 and current < total:
            return
        last_time["t"] = now
        elapsed = now - start
        fps = current / elapsed if elapsed > 0 else 0.0
        if total > 0:
            pct = 100.0 * current / total
            eta = (total - current) / fps if fps > 0 else 0.0
            line = (f"  {current}/{total} frames ({pct:5.1f}%)  "
                    f"{fps:5.1f} fps  ETA {eta:5.1f}s")
        else:
            line = f"  frame {current}  {fps:5.1f} fps"
        if is_tty:
            print(f"\r{line}", end="", flush=True)
        else:
            print(line, flush=True)

    return cb


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZebraGuard 違規偵測 CLI")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--roi", required=True, type=Path)
    parser.add_argument("--homography", required=True, type=Path)

    out = parser.add_mutually_exclusive_group()
    out.add_argument("--output", type=Path, help="輸出扁平 JSON(除錯)")
    out.add_argument("--project", type=Path, help="寫入 .zgproj 專案(覆蓋違規清單,新建或續用)")

    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--stride", type=int, default=1, help="vid_stride;每 N 幀取 1 幀")
    parser.add_argument("--yield-distance", type=float, default=3.0, help="未禮讓距離閾值 (m)")
    parser.add_argument(
        "--stop-threshold", type=float, default=5.0, help="視為停讓的速度 (km/h)"
    )
    parser.add_argument("--conf", type=float, default=0.3, help="偵測 confidence 閾值")
    return parser.parse_args(argv)


def _config_to_dict(cfg: PipelineConfig) -> dict:
    """PipelineConfig → JSON-serialisable dict(frozenset 要轉 list)。"""
    d = asdict(cfg)
    d["vehicle_classes"] = sorted(cfg.vehicle_classes)
    return d


def _write_flat_json(
    output: Path, result: PipelineResult, config: PipelineConfig
) -> None:
    data = {
        "video": {
            "path": str(result.video.path),
            "fps": result.video.fps,
            "frame_count": result.video.frame_count,
            "width": result.video.width,
            "height": result.video.height,
            "duration_sec": result.video.duration_sec,
        },
        "config": _config_to_dict(config),
        "violations": [v.to_json() for v in result.violations],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"結果寫入 JSON: {output}")


def _write_project(
    project_path: Path,
    video_path: Path,
    roi: RoiPolygon,
    homography: Homography,
    result: PipelineResult,
    config: PipelineConfig,
) -> None:
    """新建或載入 Project,寫入 ROI/homography/config/violations。"""
    if project_path.exists():
        print(f"載入既有專案: {project_path}")
        project = Project.load(project_path)
    else:
        print(f"建立新專案: {project_path}")
        project = Project.create(project_path, video_path)

    with project:
        project.save_roi(roi.to_json())
        project.save_homography(homography.to_json())
        project.save_config(_config_to_dict(config))
        project.save_violations(result.violations)
        project.update_progress("analyzed", violations=len(result.violations))
    print(f"結果寫入專案: {project_path}")
    print(f"  - {len(result.violations)} 筆違規候選於 project.db")
    print(f"  - ROI / homography / config 於 project.json")


def run_pipeline_main(argv: list[str] | None = None) -> int:
    _enable_line_buffering()
    args = _parse_args(argv)

    if not args.video.exists():
        print(f"找不到影片: {args.video}", file=sys.stderr)
        return 2

    if args.output is None and args.project is None:
        # 預設:寫扁平 JSON 到 violations.json
        args.output = Path("violations.json")

    print(f"載入 ROI: {args.roi}")
    roi = RoiPolygon.load(args.roi)
    print(f"  {len(roi.points)} 個頂點")

    print(f"載入 Homography: {args.homography}")
    homography = Homography.load(args.homography)

    config = PipelineConfig(
        yield_distance_m=args.yield_distance,
        stop_threshold_kmh=args.stop_threshold,
        detection_conf=args.conf,
        vid_stride=args.stride,
        model_name=args.model,
    )

    print(f"開始分析 {args.video}...")
    start = time.monotonic()
    result = run_pipeline(
        video_path=args.video,
        roi=roi,
        homography=homography,
        config=config,
        on_progress=_build_progress_printer(),
    )
    elapsed = time.monotonic() - start

    print()
    print(f"完成,耗時 {elapsed:.1f}s")
    print(
        f"影片:{result.video.fps:.2f} fps, {result.video.frame_count} frames, "
        f"{result.video.duration_sec:.1f}s"
    )
    print(f"追蹤:{result.n_tracks} tracks")
    print(f"找到 {len(result.violations)} 筆違規候選")

    if args.project is not None:
        _write_project(args.project, args.video, roi, homography, result, config)
    else:
        _write_flat_json(args.output, result, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline_main())
