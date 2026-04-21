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
from zebraguard.ml.exceptions import Cancelled
from zebraguard.utils.paths import presets_dir, scripts_dir


def _load_pipeline_module(script_name: str, module_name: str):
    """以 importlib 載入 scripts/ 下的 pipeline script。"""
    path = scripts_dir() / script_name
    if not path.is_file():
        raise FileNotFoundError(f"找不到 pipeline script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 {module_name} spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_zero_shot_module():
    return _load_pipeline_module("zero_shot_detect.py", "zero_shot_detect")


def _load_static_parking_module():
    return _load_pipeline_module("static_parking_detect.py", "static_parking_detect")


_BACKEND_PRESET = {
    "mask2former": "v7_baseline_params.json",
    "yolo_seg":    "yolo_seg_initial.json",  # 最原版 = v7 閾值 + ysc 0.25
}


def _load_preset(backend: str) -> dict[str, Any]:
    """依 crosswalk backend 讀對應的 preset。"""
    fname = _BACKEND_PRESET.get(backend, "v7_baseline_params.json")
    with open(presets_dir() / fname, encoding="utf-8") as f:
        data = json.load(f)
    # 去掉僅供人看的注釋欄位,免得後面 dict-sprinkle 進去時卡住
    return {k: v for k, v in data.items() if not k.startswith("_")}


# project.meta.pipeline_config 存檔時夾帶的中繼鍵(不是 run() 接受的參數,
# 過濾掉才能把 dict 直接攤平成 kwargs)
_CONFIG_METADATA_KEYS = {"preset", "mode", "crosswalk_backend"}


def _resolve_params(project: Project, backend: str) -> dict[str, Any]:
    """優先用 project 自己存的 pipeline_config(使用者在進階設定動過),
    否則 fallback 到 backend 對應的 preset。"""
    saved = project.meta.pipeline_config or {}
    if saved:
        return {k: v for k, v in saved.items() if k not in _CONFIG_METADATA_KEYS}
    return _load_preset(backend)


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
        project = Project.load(self._project_path)
        try:
            video_path = Path(project.meta.video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"影片不存在: {video_path}")

            mode = project.meta.mode or "dashcam"

            if mode == "static":
                self._run_static(project, video_path)
                return

            self._run_dashcam(project, video_path)
        finally:
            project.close()

    def _run_dashcam(self, project: Project, video_path: Path) -> None:
        """原有 mask2former / yolo_seg 未禮讓檢測流程。"""
        zs = _load_zero_shot_module()

        backend = project.meta.crosswalk_backend or "mask2former"
        params = _resolve_params(project, backend)
        if self._max_seconds is not None:
            params["max_seconds"] = float(self._max_seconds)

        if backend == "yolo_seg":
            weights = project.meta.yolo_seg_weights
            if not weights:
                raise RuntimeError(
                    "專案 crosswalk_backend=yolo_seg 但未指定 yolo_seg_weights;"
                    "請在新專案 wizard 指定 .pt 路徑。"
                )
            self.log_line.emit(f"載入 YOLO-seg 斑馬線權重:{weights}")
        else:
            self.log_line.emit("載入 Mask2Former 模型(首次約需 30 秒)…")

        params_snapshot = {
            "preset": _BACKEND_PRESET.get(backend, "v7_baseline_params.json")
                .replace("_params.json", "").replace(".json", ""),
            "mode": "dashcam",
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
                yolo_seg_classes=None,
                yolo_seg_conf=float(params.get("yolo_seg_conf", 0.25)),
                yolo_seg_imgsz=int(params.get("yolo_seg_imgsz", 640)),
                yolo_seg_min_component_px=int(
                    params.get("yolo_seg_min_component_px", 200)
                ),
                progress_cb=cb,
                cancel_event=self._cancel,
            )
        except Cancelled:
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

    def _run_static(self, project: Project, video_path: Path) -> None:
        """違停檢測 pipeline(靜態攝影機 mode)。"""
        sp = _load_static_parking_module()

        no_parking_zones = list(project.meta.no_parking_zones or [])
        if not no_parking_zones:
            self.log_line.emit(
                "警告:沒有畫違法區 → 不會有候選事件(預設保守)。"
            )

        # 閾值優先讀 project.meta(使用者在 ROI 編輯器可調);備援預設
        params = {
            "yolo_weights": "yolo11n.pt",
            "conf": 0.3,
            "imgsz": 640,
            "vid_stride": 1,
            "stopped_threshold_sec": float(project.meta.stopped_threshold_sec or 60.0),
            "stopped_max_displacement_px": float(project.meta.stopped_max_displacement_px or 20.0),
            "roi_overlap_threshold": 0.5,
            "strip_frac": 0.20,
            "vehicle_classes": [2, 5, 7],
        }
        if self._max_seconds is not None:
            params["max_seconds"] = float(self._max_seconds)

        params_snapshot = {
            "preset": "static_default",
            "mode": "static",
            **params,
            "n_no_parking_zones": len(no_parking_zones),
        }
        project.save_config(params_snapshot)
        project.update_progress("analyzing", processed_frames=0)

        self.log_line.emit(
            f"載入 YOLO + ByteTrack…"
            f" 停留閾值 {params['stopped_threshold_sec']:.0f}s /"
            f" 位移 {params['stopped_max_displacement_px']:.0f}px"
        )

        def cb(stage: str, current: int, total: int, hits: int) -> None:
            self.progress.emit(stage, int(current), int(total), int(hits))

        try:
            report = sp.run(
                video_path,
                no_parking_zones=no_parking_zones,
                yolo_weights=params["yolo_weights"],
                conf=params["conf"],
                imgsz=params["imgsz"],
                vid_stride=params["vid_stride"],
                stopped_threshold_sec=params["stopped_threshold_sec"],
                stopped_max_displacement_px=params["stopped_max_displacement_px"],
                roi_overlap_threshold=params["roi_overlap_threshold"],
                strip_frac=params["strip_frac"],
                vehicle_classes=params["vehicle_classes"],
                max_seconds=params.get("max_seconds"),
                output_json=None,
                progress_cb=cb,
                cancel_event=self._cancel,
            )
        except Cancelled:
            project.update_progress("cancelled")
            self.cancelled.emit()
            return

        # 把 ParkingEvent candidates 轉成 events 表的格式
        candidates = report.get("candidates", [])
        events_rows: list[dict] = []
        for c in candidates:
            events_rows.append({
                "start_frame": c["start_frame"],
                "end_frame": c["end_frame"],
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "min_distance_px": 0.0,
                "peak_speed_px": 0.0,
                "ped_track_ids": [],
                "veh_track_ids": [c["vehicle_track_id"]],
            })
        project.save_events(events_rows)
        project.update_progress(
            "done",
            processed_frames=report.get("frame_count", 0),
            events=len(events_rows),
        )
        self.done.emit(report)


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
