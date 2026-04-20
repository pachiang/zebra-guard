"""YOLO-seg 斑馬線分割 — CrosswalkSource 實作(Dashcam mode 的 CPU-friendly 路線)。

設計:
  · 使用 Ultralytics `.pt` 權重(多半在 Roboflow / HuggingFace 上可找到預訓練,
    或使用者自行 fine-tune)。權重**不綁附於 repo**;使用者透過 config 指定路徑。
  · 單類別模型(模型只認得 "crosswalk" 一種類別)預設接受所有輸出。
    多類別模型(例如連帶辨識 stop line、sidewalk)用 `class_names` 指定要保留哪些。
  · YOLO-seg 推論速度 CPU 可接受,預設每幀都跑(`infer_every=1`);
    若要省成本可調大。

對比 Mask2FormerSource:
  · 不需 GPU(但有 GPU 會更快)
  · 模型小(.pt 通常 6-50 MB)
  · 精度取決於使用的權重;差的權重會漏掉斑馬線、或誤把停止線當斑馬線
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from zebraguard.ml.crosswalk.base import LabeledMask, dilate_and_label


@dataclass(slots=True)
class YoloSegConfig:
    weights: str = ""                          # 必填:.pt 路徑或模型名(ultralytics 會自動下載)
    class_names: list[str] | None = None       # None = 接受所有類別
    conf: float = 0.25
    imgsz: int = 640
    dilate_px: int = 20
    min_component_px: int = 200
    min_mask_area_frac: float = 0.005
    infer_every: int = 1                       # 每 N 幀推論一次;中間重用上次結果
    device: str = "auto"                       # "auto" / "cpu" / "cuda" / "0"


class YoloSegSource:
    """Dashcam backend:YOLO-seg 權重提供斑馬線 mask。"""

    def __init__(self, config: YoloSegConfig) -> None:
        if not config.weights:
            raise ValueError("YoloSegConfig.weights 必須指定 .pt 路徑或模型名")
        self.config = config
        self._model = None
        self._allowed_cls_ids: set[int] | None = None

        self._cached_labels: LabeledMask | None = None
        self._last_infer_frame: int = -(10**9)
        self._cached_frame_shape: tuple[int, int] | None = None

    # ---- lifecycle --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        weights = self.config.weights
        # 若是相對路徑且不存在,讓 ultralytics 自行處理(它會嘗試從 URL 或 hub 拉)
        if Path(weights).suffix == ".pt" and Path(weights).exists():
            weights = str(Path(weights).resolve())
        self._model = YOLO(weights)

        if self.config.class_names is not None:
            # ultralytics model.names 是 {id: name}
            name_to_id = {name: cid for cid, name in self._model.names.items()}
            missing = [n for n in self.config.class_names if n not in name_to_id]
            if missing:
                raise ValueError(
                    f"model 中找不到類別 {missing};可用類別:{list(name_to_id.keys())}"
                )
            self._allowed_cls_ids = {name_to_id[n] for n in self.config.class_names}
        else:
            self._allowed_cls_ids = None  # 接受所有類別

    def close(self) -> None:
        self._model = None

    # ---- segmentation -----------------------------------------------------

    def _segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """回傳原尺寸的 uint8 binary mask(0/255),合併所有 crosswalk 實例。"""
        assert self._model is not None
        h, w = frame_bgr.shape[:2]
        device = None if self.config.device == "auto" else self.config.device
        results = self._model.predict(
            frame_bgr,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            device=device,
            verbose=False,
        )
        binary = np.zeros((h, w), dtype=np.uint8)
        if not results:
            return binary
        r = results[0]
        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is None or masks.xy is None or len(masks.xy) == 0:
            return binary

        # 過濾類別
        keep: list[int] = list(range(len(masks.xy)))
        if self._allowed_cls_ids is not None and boxes is not None and boxes.cls is not None:
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            keep = [i for i in keep if int(cls_arr[i]) in self._allowed_cls_ids]

        for i in keep:
            poly = masks.xy[i]
            if poly is None or len(poly) < 3:
                continue
            cv2.fillPoly(binary, [poly.astype(np.int32)], 255)
        return binary

    # ---- CrosswalkSource interface ---------------------------------------

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> LabeledMask:
        h, w = frame_bgr.shape[:2]
        if self._cached_frame_shape is not None and self._cached_frame_shape != (h, w):
            self._cached_labels = None

        need_refresh = (
            self._cached_labels is None
            or (frame_idx - self._last_infer_frame) >= self.config.infer_every
        )
        if need_refresh:
            self._ensure_loaded()
            raw = self._segment(frame_bgr)
            area = int((raw > 0).sum())
            min_area_px = int(self.config.min_mask_area_frac * h * w)
            if area < min_area_px:
                self._cached_labels = np.zeros((h, w), dtype=np.int32)
            else:
                self._cached_labels = dilate_and_label(
                    raw,
                    dilate_px=self.config.dilate_px,
                    min_component_px=self.config.min_component_px,
                )
            self._cached_frame_shape = (h, w)
            self._last_infer_frame = frame_idx
        assert self._cached_labels is not None
        return self._cached_labels
