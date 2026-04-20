"""Crosswalk source:每幀提供斑馬線位置的插拔式 backend。

兩種 mode × N 個 backend:
  - Static mode  → StaticRoiSource(使用者手繪 polygon)
  - Dashcam mode → Mask2FormerSource / YoloSegSource / 未來的其他實作

介面定義在 `base.py`;下游(規則引擎、聚合、UI)共用同一份程式碼,只透過
CrosswalkSource 取得每幀的 labeled mask。

詳見 `docs/pipelines.md`。
"""

from zebraguard.ml.crosswalk.base import (
    CrosswalkSource,
    LabeledMask,
    NullCrosswalkSource,
    dilate_and_label,
)

__all__ = [
    "CrosswalkSource",
    "LabeledMask",
    "NullCrosswalkSource",
    "dilate_and_label",
]
