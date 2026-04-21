"""進階設定 dialog — 讓使用者調整 pipeline 參數並觸發重跑。

放常用的幾個;ego / rider filter 等不常動的就讓它留在 preset 預設值。

設計:
  · 頂部「預設」下拉:選 preset 會立刻 populate 所有欄位;任何欄位改動會切
    到「自訂」
  · backend 顯示為 read-only(切 backend 代表整個管線變,建議新建專案)
  · 關掉改動不會存;要按「套用並重跑」才 persist 到 project.meta.pipeline_config

返回:
  · dialog.result_params()  → 套用的參數 dict(含 __rerun__ = True 表示要重跑)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from zebraguard.utils.paths import presets_dir

_CUSTOM_NAME = "自訂"

# 欄位定義:(key, label, type, min, max, step, default_value, help_text)
# type: "int" | "float" | "percent";default 是「某 preset 沒這個欄位時」fallback 用
_COMMON_FIELDS: list[tuple[str, str, str, float, float, float, float, str]] = [
    ("mask_every", "偵測頻率(每 N 幀重跑)", "int", 1, 30, 1, 5,
     "越小越即時但越耗 CPU。yolo_seg 建議 1-3,mask2former 建議 5。"),
    ("min_mask_area_frac", "斑馬線最小面積(% 畫面)", "percent", 0.05, 5.0, 0.05, 0.005,
     "低於此比例的 mask 會被當雜訊丟;調小可抓遠處小斑馬線但易誤判。"),
    ("dilate_px", "斑馬線膨脹像素", "int", 0, 60, 1, 20,
     "mask 膨脹讓車輛 bbox 底部 strip 更容易壓到;同一路口不同斑馬線靠此區分。"),
    ("moving_px", "車輛位移閾值 (px/幀)", "float", 0.0, 20.0, 0.5, 2.0,
     "中心位移 > 此值才算車輛在移動;<= 此值視為停讓不觸發。"),
    ("conf", "YOLO 偵測信心", "float", 0.05, 0.8, 0.05, 0.2,
     "person / vehicle 偵測閾值。降低可抓更多小 bbox 但易誤判。"),
    ("person_conf", "行人最低信心", "float", 0.1, 0.9, 0.05, 0.35,
     "額外對 person 類別拉高;避開把機車誤認為行人的低信心 bbox。"),
    ("merge_gap_sec", "事件合併間隙 (秒)", "float", 0.0, 5.0, 0.1, 0.6,
     "連續 hit 間隔若小於此秒數會被聚成一筆事件。"),
    ("min_event_frames", "最短事件幀數", "int", 1, 30, 1, 2,
     "短於此幀數的事件當作閃爍誤判丟棄。"),
]

# 只在 yolo_seg backend 顯示的欄位
_YOLO_SEG_FIELDS: list[tuple[str, str, str, float, float, float, float, str]] = [
    ("yolo_seg_conf", "YOLO-seg 斑馬線信心", "float", 0.05, 0.8, 0.05, 0.25,
     "斑馬線模型的偵測閾值。降低可抓遠處小斑馬線。"),
    ("yolo_seg_imgsz", "YOLO-seg 輸入解析度", "int", 320, 1280, 64, 640,
     "模型內部 resize 的短邊像素;愈高細節愈多但 2-3 倍慢。"),
    ("yolo_seg_min_component_px", "斑馬線 min component (px)", "int", 10, 1000, 10, 200,
     "膨脹後小於此像素的斑馬線塊丟棄。"),
]


def _make_spin(
    spec: tuple[str, str, str, float, float, float, float, str]
) -> QSpinBox | QDoubleSpinBox:
    _, _, typ, lo, hi, step, _default, _ = spec
    if typ == "int":
        sb = QSpinBox()
        sb.setRange(int(lo), int(hi))
        sb.setSingleStep(int(step))
    elif typ == "percent":
        sb = QDoubleSpinBox()
        sb.setDecimals(3)
        sb.setRange(float(lo), float(hi))
        sb.setSingleStep(float(step))
        sb.setSuffix(" %")
    else:  # float
        sb = QDoubleSpinBox()
        sb.setDecimals(2)
        sb.setRange(float(lo), float(hi))
        sb.setSingleStep(float(step))
    return sb


def _param_to_field(key: str, value: Any, typ: str) -> float | int:
    """將 pipeline_config 裡的值轉成 spinbox 接受的數字(percent 類要 × 100)。"""
    if value is None:
        return 0
    if typ == "percent":
        return float(value) * 100.0
    if typ == "int":
        return int(value)
    return float(value)


def _field_to_param(field_value: float, typ: str) -> float | int:
    if typ == "percent":
        return round(float(field_value) / 100.0, 5)
    if typ == "int":
        return int(field_value)
    return round(float(field_value), 3)


class AdvancedSettingsDialog(QDialog):
    """回傳 result:None = 取消;dict = 參數(有 __rerun__ 鍵表示要重跑)。"""

    def __init__(
        self,
        current_config: dict[str, Any],
        backend: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("進階設定")
        self.setModal(True)
        self.setMinimumSize(560, 560)
        self._backend = backend
        self._original = copy.deepcopy(current_config) or {}
        self._result: dict[str, Any] | None = None
        self._suppress_change = False

        self._build_ui()
        # 填入目前值
        self._load_values(self._original, select_preset="自訂")
        self._try_match_preset()

    # ---- public -------------------------------------------------

    def result_params(self) -> dict[str, Any] | None:
        return self._result

    # ---- layout -------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 18)
        outer.setSpacing(12)

        title = QLabel("進階設定")
        tf = QFont()
        tf.setPointSize(16)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)

        sub = QLabel(
            f"Backend:<b>  {self._backend}</b>"
            "  &nbsp;·&nbsp;  "
            "調整後按『套用並重跑』會清除目前所有事件與審查狀態"
        )
        sub.setTextFormat(Qt.TextFormat.RichText)
        sub.setObjectName("Hint")
        sub.setWordWrap(True)

        outer.addWidget(title)
        outer.addWidget(sub)

        # Preset picker
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(QLabel("預設組合:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(_CUSTOM_NAME)
        # 列出所有 *_params.json 或 *_baseline.json
        for p in sorted(presets_dir().glob("*.json")):
            self.preset_combo.addItem(p.stem)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, stretch=1)
        outer.addLayout(preset_row)

        # Scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        form_wrap = QFrame()
        form_wrap.setObjectName("Card")
        form_wrap.setProperty("class", "card")
        form = QFormLayout(form_wrap)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # (type, default, spinbox)
        self._fields: dict[str, tuple[str, float, QSpinBox | QDoubleSpinBox]] = {}

        def _add_section(label_text: str) -> None:
            sep = QLabel(label_text)
            sep.setObjectName("MetricLabel")
            form.addRow(sep)

        def _add_spec(
            spec: tuple[str, str, str, float, float, float, float, str],
        ) -> None:
            key, label, typ, _lo, _hi, _step, default, hint = spec
            sb = _make_spin(spec)
            sb.valueChanged.connect(self._on_field_changed)
            sb.setToolTip(hint)
            form.addRow(label, sb)
            self._fields[key] = (typ, default, sb)

        _add_section("偵測 / 頻率")
        for spec in _COMMON_FIELDS:
            if spec[0] in {"mask_every", "min_mask_area_frac", "dilate_px"}:
                _add_spec(spec)

        _add_section("違規判定")
        for spec in _COMMON_FIELDS:
            if spec[0] in {"moving_px", "merge_gap_sec", "min_event_frames"}:
                _add_spec(spec)

        _add_section("偵測信心閾值")
        for spec in _COMMON_FIELDS:
            if spec[0] in {"conf", "person_conf"}:
                _add_spec(spec)

        if self._backend == "yolo_seg":
            _add_section("YOLO-seg 斑馬線模型")
            for spec in _YOLO_SEG_FIELDS:
                _add_spec(spec)

        scroll.setWidget(form_wrap)
        outer.addWidget(scroll, stretch=1)

        # Buttons
        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setProperty("ghost", True)
        self.cancel_btn.clicked.connect(self.reject)

        self.reset_btn = QPushButton("重置為預設")
        self.reset_btn.clicked.connect(self._reset_to_backend_preset)

        self.apply_btn = QPushButton("套用並重跑")
        self.apply_btn.setProperty("accent", True)
        self.apply_btn.setMinimumWidth(140)
        self.apply_btn.clicked.connect(self._on_apply)

        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.reset_btn)
        btns.addStretch(1)
        btns.addWidget(self.apply_btn)
        outer.addLayout(btns)

    # ---- slots --------------------------------------------------

    def _on_preset_changed(self, name: str) -> None:
        if self._suppress_change:
            return
        if name == _CUSTOM_NAME:
            return
        preset_path = presets_dir() / f"{name}.json"
        if not preset_path.is_file():
            return
        with open(preset_path, encoding="utf-8") as f:
            preset = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        # 鎖住 change signal,免得 load 觸發 combo 切回「自訂」
        self._load_values(preset, select_preset=name)

    def _on_field_changed(self, _new_val) -> None:  # noqa: ANN001
        if self._suppress_change:
            return
        # 使用者手動改 → 切到「自訂」
        if self.preset_combo.currentText() != _CUSTOM_NAME:
            self._suppress_change = True
            self.preset_combo.setCurrentText(_CUSTOM_NAME)
            self._suppress_change = False

    def _reset_to_backend_preset(self) -> None:
        fname = "yolo_seg_baseline" if self._backend == "yolo_seg" else "v7_baseline_params"
        idx = self.preset_combo.findText(fname)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _on_apply(self) -> None:
        # 組裝 pipeline_config:以 current_config 為底,覆蓋 dialog 內欄位
        out = copy.deepcopy(self._original)
        for key, (typ, _default, sb) in self._fields.items():
            out[key] = _field_to_param(sb.value(), typ)
        out["__rerun__"] = True
        out["preset"] = self.preset_combo.currentText()
        self._result = out
        self.accept()

    # ---- helpers ------------------------------------------------

    def _load_values(self, config: dict[str, Any], select_preset: str) -> None:
        """把 config 的值填入 dialog 欄位;**缺的欄位一律重設為該欄位的 default**,
        避免先選 preset A 再選 preset B 時殘留 A 的值看起來 preset 切換沒生效。"""
        self._suppress_change = True
        try:
            for key, (typ, default, sb) in self._fields.items():
                value = config.get(key, default)
                sb.setValue(_param_to_field(key, value, typ))
            idx = self.preset_combo.findText(select_preset)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
        finally:
            self._suppress_change = False

    def _try_match_preset(self) -> None:
        """若目前值恰好完全等於某個 preset,把 combo 對應到那個 preset。"""
        for i in range(self.preset_combo.count()):
            name = self.preset_combo.itemText(i)
            if name == _CUSTOM_NAME:
                continue
            p = presets_dir() / f"{name}.json"
            if not p.is_file():
                continue
            with open(p, encoding="utf-8") as f:
                preset = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
            if all(
                _compare_val(
                    self._original.get(key, default),
                    preset.get(key, default),
                    typ,
                )
                for key, (typ, default, _) in self._fields.items()
            ):
                self._suppress_change = True
                self.preset_combo.setCurrentText(name)
                self._suppress_change = False
                return


def _compare_val(a: Any, b: Any, typ: str) -> bool:
    if a is None or b is None:
        return a == b
    if typ == "int":
        return int(a) == int(b)
    return abs(float(a) - float(b)) < 1e-6
