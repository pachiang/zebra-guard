"""CrosswalkSource Protocol + 共用 post-processing。

設計意圖詳見 `docs/pipelines.md`。重點:
  · Source 輸出 labeled mask(int32, 0=bg, 1..N=components)
  · Component 編號在同一幀內唯一;跨幀**不保證穩定**——不同幀的同一片斑馬線
    可能拿到不同的 id。下游的「同一 component 規則」只看同一幀內的 id 是否相等,
    不追蹤跨幀 id。
  · Source 自行決定快取(每幀跑 / 每 N 幀跑 / 整段影片固定)
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

# Labeled mask:shape (H, W) int32
# 0           = background
# 1, 2, 3, …  = different crosswalk components on this frame
LabeledMask = np.ndarray


class CrosswalkSource(Protocol):
    """每幀斑馬線位置的來源。

    Lifecycle:
      source = SomeCrosswalkSource(...)
      for frame_idx, frame in video:
          labels = source.get_labels(frame, frame_idx)  # 下游處理
      source.close()
    """

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> LabeledMask:
        """回傳 shape=(H, W) int32 labeled mask。H, W 必須等於 frame 的尺寸。

        若該幀沒有偵測到合格的斑馬線(例如 mask 過小),回傳全 0 即可;
        下游的「bbox 是否壓到 component」判定會自動不觸發。
        """
        ...

    def close(self) -> None:
        """釋放 GPU / 檔案 / 模型資源。不得在 close 之後再呼叫 get_labels。"""
        ...


def dilate_and_label(
    binary_mask: np.ndarray,
    dilate_px: int,
    min_component_px: int = 200,
) -> LabeledMask:
    """Source 實作共用的 post-processing:膨脹 → connected-components → 去小碎片。

    為什麼要膨脹後才 label:相鄰的斑馬線在原始 mask 上可能斷開,膨脹後合成一塊
    (同一路口的一條斑馬線);不同路口的斑馬線距離夠遠,膨脹後仍分開。這是
    v7 的「同一 component 規則」能區分「兩側斑馬線」的基礎。

    `min_component_px` = 200 與 `scripts/zero_shot_detect.py` 的舊行為一致。
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        dilated = cv2.dilate(binary_mask, k)
    else:
        dilated = binary_mask

    n_labels, labels = cv2.connectedComponents(dilated)
    if n_labels <= 1:
        return np.zeros(binary_mask.shape, dtype=np.int32)

    kept = np.zeros(n_labels, dtype=np.int32)
    next_id = 1
    for lab in range(1, n_labels):
        if int((labels == lab).sum()) >= min_component_px:
            kept[lab] = next_id
            next_id += 1
    return kept[labels].astype(np.int32)


class NullCrosswalkSource:
    """永遠回傳全 0 的 source。

    用途:
      · 煙霧測試 pipeline 其餘部分(不載入模型)
      · 未初始化時的 safe default
    """

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> LabeledMask:
        h, w = frame_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.int32)

    def close(self) -> None:
        pass
