"""Mask2Former(Mapillary Vistas)自動斑馬線分割 — CrosswalkSource 實作。

對應 `scripts/zero_shot_detect.py` 中的 `load_mask2former` + `segment_zebra`
+ 膨脹 + connected-components 那整個區段。抽到這裡以便:
  1. Dashcam mode 的其他 backend(YOLO-seg 等)走同一下游
  2. UI / worker 可透過統一介面注入不同 backend

本 backend 需要 GPU(swin-large 在 CPU 上的延遲極高)。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from zebraguard.ml.crosswalk.base import LabeledMask, dilate_and_label

# Mapillary Vistas v1.2 類別:swin-large 模型有兩個相關 label
ZEBRA_LABELS = ("Crosswalk - Plain", "Lane Marking - Crosswalk")


@dataclass(slots=True, frozen=True)
class Mask2FormerConfig:
    model_name: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    mask_every: int = 5          # 每 N 個有被處理的幀重跑一次 Mask2Former
    dilate_px: int = 20
    mask_imgsz: int = 640        # > 0 則推論時短邊 resize 至此;0 = 原尺寸
    min_mask_area_frac: float = 0.005  # 小於此比例的 mask 視為雜訊丟棄
    min_component_px: int = 200  # 膨脹後小於此像素的 component 丟棄


class Mask2FormerSource:
    """Dashcam backend:Mask2Former 每 N 幀自動分割斑馬線。"""

    def __init__(
        self,
        config: Mask2FormerConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or Mask2FormerConfig()
        self._device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor = None
        self._model = None
        self._zebra_ids: list[int] = []

        # 快取
        self._cached_labels: LabeledMask | None = None
        self._last_seg_frame: int = -(10**9)
        self._cached_frame_shape: tuple[int, int] | None = None

    # ---- lifecycle --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self._processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        self._model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(self.config.model_name)
            .to(self._device)
            .eval()
        )
        id2label = self._model.config.id2label
        self._zebra_ids = [int(i) for i, name in id2label.items() if name in ZEBRA_LABELS]
        if not self._zebra_ids:
            raise RuntimeError(
                f"None of {ZEBRA_LABELS} found in model labels "
                f"({len(id2label)} classes). Check model version."
            )

    def close(self) -> None:
        self._model = None
        self._processor = None
        # 讓 GC 決定何時釋放 GPU memory
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- segment ----------------------------------------------------------

    def _segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """回傳原尺寸的 uint8 binary mask(0/255)。"""
        assert self._processor is not None and self._model is not None
        h_full, w_full = frame_bgr.shape[:2]
        src = frame_bgr
        size = self.config.mask_imgsz
        if size > 0 and min(h_full, w_full) > size:
            if h_full <= w_full:
                new_h = size
                new_w = int(round(w_full * size / h_full))
            else:
                new_w = size
                new_h = int(round(h_full * size / w_full))
            src = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target_size = pil.size[::-1]
        seg = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size]
        )[0]
        seg_np = seg.cpu().numpy().astype(np.int32)
        mask_small = np.isin(seg_np, self._zebra_ids).astype(np.uint8) * 255
        if mask_small.shape != (h_full, w_full):
            return cv2.resize(
                mask_small, (w_full, h_full), interpolation=cv2.INTER_NEAREST
            )
        return mask_small

    # ---- CrosswalkSource interface ---------------------------------------

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> LabeledMask:
        h, w = frame_bgr.shape[:2]
        # 影片尺寸變了(理論上不會,但保險起見)→ 丟快取
        if self._cached_frame_shape is not None and self._cached_frame_shape != (h, w):
            self._cached_labels = None

        need_refresh = (
            self._cached_labels is None
            or (frame_idx - self._last_seg_frame) >= self.config.mask_every
        )
        if need_refresh:
            self._ensure_loaded()
            raw = self._segment(frame_bgr)
            area = int((raw > 0).sum())
            min_area_px = int(self.config.min_mask_area_frac * h * w)
            if area < min_area_px:
                # v7 的決定:小面積 mask 棄用,避免在 dashcam 移動下殘留 ghost
                self._cached_labels = np.zeros((h, w), dtype=np.int32)
            else:
                self._cached_labels = dilate_and_label(
                    raw,
                    dilate_px=self.config.dilate_px,
                    min_component_px=self.config.min_component_px,
                )
            self._cached_frame_shape = (h, w)
            self._last_seg_frame = frame_idx
        assert self._cached_labels is not None
        return self._cached_labels
