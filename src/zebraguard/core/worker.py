"""QThread worker:於背景執行 v7 zero-shot pipeline,以 Qt signal 回報進度。

設計決策:
- `scripts/zero_shot_detect.py` 是主要的 pipeline 實作。為了避免把整個檔案搬進
  zebraguard package(會破壞 CLI 的 ROOT 計算與 presets 路徑),此處改以
  importlib 動態載入模組;打包時 PyInstaller 的 datas 需涵蓋 scripts/。
- v7 參數固定從 `scripts/presets/v7_baseline_params.json` 載入,不開放 UI 微調
  (見 MVP P1 清單再決定何時開放)。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from zebraguard.core.project import Project
from zebraguard.utils.paths import presets_dir, scripts_dir


def _load_zero_shot_module():
    """以 importlib 載入 scripts/zero_shot_detect.py,避免污染 sys.path。"""
    path = scripts_dir() / "zero_shot_detect.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 zero_shot_detect.py: {path}")
    spec = importlib.util.spec_from_file_location("zero_shot_detect", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 zero_shot_detect spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("zero_shot_detect", module)
    spec.loader.exec_module(module)
    return module


def _load_v7_params() -> dict[str, Any]:
    """讀 v7_baseline_params.json;UI 全流程固定這組參數。"""
    path = presets_dir() / "v7_baseline_params.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class PipelineWorker(QObject):
    """v7 pipeline 背景執行器。

    使用方式:
      worker = PipelineWorker(project_path)
      thread = QThread()
      worker.moveToThread(thread)
      thread.started.connect(worker.run)
      worker.finished.connect(thread.quit)
      # 連接 progress / done / error / cancelled 到 UI
      thread.start()
    """

    # (stage, current, total, hits)
    # stage ∈ {"loading_mask", "loading_yolo", "analyzing", "done"}
    progress = Signal(str, int, int, int)

    # 管線執行完成,附帶事件清單 (來自 zero_shot_detect 的 report dict)
    done = Signal(dict)

    # 使用者主動取消
    cancelled = Signal()

    # 非預期錯誤,附 traceback 字串
    error = Signal(str)

    # 進度訊息(給 log 區塊顯示)
    log_line = Signal(str)

    # Lifecycle
    finished = Signal()

    def __init__(self, project_path: Path, max_seconds: float | None = None) -> None:
        super().__init__()
        self._project_path = Path(project_path)
        self._max_seconds = max_seconds
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            self._run_inner()
        except Exception:  # noqa: BLE001
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    def _run_inner(self) -> None:
        zs = _load_zero_shot_module()
        cancelled_exc = zs.Cancelled

        project = Project.load(self._project_path)
        try:
            video_path = Path(project.meta.video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"影片不存在: {video_path}")

            mode = project.meta.mode or "dashcam"
            backend = project.meta.crosswalk_backend or "mask2former"
            if mode == "static":
                raise NotImplementedError(
                    "Static mode 尚未實作(見 docs/pipelines.md)"
                )

            params = _load_v7_params()
            if self._max_seconds is not None:
                params["max_seconds"] = float(self._max_seconds)

            if backend == "yolo_seg":
                weights = project.meta.yolo_seg_weights
                if not weights:
                    raise RuntimeError(
                        "專案 crosswalk_backend=yolo_seg 但未指定 yolo_seg_weights;"
                        "請在新專案 wizard 指定 .pt 路徑。"
                    )
                self.log_line.emit(
                    f"載入 YOLO-seg 斑馬線權重:{weights}"
                )
            else:
                self.log_line.emit("載入 Mask2Former 模型(首次約需 30 秒)…")

            params_snapshot = {
                "preset": "v7_baseline",
                "mode": mode,
                "crosswalk_backend": backend,
                **params,
            }
            project.save_config(params_snapshot)
            project.update_progress("analyzing", processed_frames=0)

            import torch

            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_line.emit(f"運算裝置:{device_str} / backend={backend}")

            def cb(stage: str, current: int, total: int, hits: int) -> None:
                self.progress.emit(stage, int(current), int(total), int(hits))

            try:
                report = zs.run(
                    video_path,
                    model_name=params["mask_model"],
                    yolo_weights=params["yolo_weights"],
                    device_str=device_str,
                    stride=params["stride"],
                    mask_every=params["mask_every"],
                    dilate_px=params["dilate_px"],
                    prox_px=params["prox_px"],
                    moving_px=params["moving_px"],
                    conf=params["conf"],
                    imgsz=params["imgsz"],
                    merge_gap_sec=params["merge_gap_sec"],
                    min_event_frames=params["min_event_frames"],
                    output_json=None,
                    preview_video=None,
                    preview_max_seconds=0.0,
                    exclude_ego=params["exclude_ego"],
                    ego_bottom_px=params["ego_bottom_px"],
                    ego_min_width_frac=params["ego_min_width_frac"],
                    ego_min_height_frac=params["ego_min_height_frac"],
                    max_seconds=params.get("max_seconds"),
                    min_mask_area_frac=params["min_mask_area_frac"],
                    annotated_out=None,
                    rider_contain_thresh=params["rider_contain_thresh"],
                    rider_foot_margin_px=params["rider_foot_margin_px"],
                    person_conf=params["person_conf"],
                    mask_imgsz=params["mask_imgsz"],
                    crosswalk_backend=backend,
                    yolo_seg_weights=project.meta.yolo_seg_weights,
                    yolo_seg_classes=None,  # 未來從 meta 讀
                    progress_cb=cb,
                    cancel_event=self._cancel,
                )
            except cancelled_exc:
                project.update_progress("cancelled")
                self.cancelled.emit()
                return

            project.save_events(report.get("events", []))
            project.update_progress(
                "done",
                processed_frames=report.get("frame_count", 0),
                events=len(report.get("events", [])),
            )
            self.done.emit(report)
        finally:
            project.close()


def start_pipeline_thread(
    project_path: Path,
    *,
    on_progress=None,
    on_log=None,
    on_done=None,
    on_cancelled=None,
    on_error=None,
    max_seconds: float | None = None,
) -> tuple[QThread, PipelineWorker]:
    """便利函式:建好 worker + thread,連好 signal,回傳未 start 的 (thread, worker)。

    呼叫端自己決定何時 `thread.start()`,以及保管 references 防止 GC。
    """
    worker = PipelineWorker(project_path, max_seconds=max_seconds)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if on_progress is not None:
        worker.progress.connect(on_progress)
    if on_log is not None:
        worker.log_line.connect(on_log)
    if on_done is not None:
        worker.done.connect(on_done)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled)
    if on_error is not None:
        worker.error.connect(on_error)

    return thread, worker
