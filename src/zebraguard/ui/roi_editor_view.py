"""違停檢測 · ROI 編輯器 — 使用者在一幀影像上畫違法區多邊形(紅黃線 / 人行道 / 禁停標誌)。

互動規範:
  · 左上「+ 新增多邊形」→ 開始繪製;畫布上 click 加頂點;double-click 或 Enter 收尾
  · 繪製中 Esc 取消;畫完自動回 idle
  · Idle:click 多邊形 body 選取(顯示 amber 描邊);Del / 右側「刪除」鍵移除
  · 頂點 drag 可移動;右鍵頂點開 context menu 刪頂點
  · 底部 slider 切換背景幀(可挑個能看清整個停車格的幀當參考)

座標系統:polygon 儲存於**影像原始像素座標**,UI 顯示時縮放置中。這避免使用者
調整視窗大小後多邊形移位。
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from zebraguard.core.project import Project

# 點擊容忍半徑(widget pixels) — 拖曳現有頂點
_HIT_RADIUS = 12
# Snap 收尾半徑 — 繪製中靠近首頂點到此距離內,點擊就收尾
_SNAP_RADIUS = 16
# 頂點半徑(idle / drawing);drawing 時放大給更強的可見度
_VERTEX_RADIUS = 6
_VERTEX_RADIUS_DRAWING = 8


class _RoiCanvas(QWidget):
    """自繪多邊形編輯畫布。"""

    polygons_changed = Signal()      # 多邊形數量 / 結構變動(新增、刪除、closure)
    selection_changed = Signal(int)  # 目前選取的 polygon index;-1 為未選
    drawing_state_changed = Signal(bool)  # True = 正在畫;False = idle

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: #000; border-radius: 8px;")

        self._pixmap: QPixmap | None = None
        self._img_size: tuple[int, int] = (0, 0)  # (w, h)

        # polygons 儲存影像座標(QPointF)
        self._polygons: list[list[QPointF]] = []
        self._drawing: list[QPointF] | None = None
        self._selected: int = -1
        self._drag: tuple[int, int] | None = None  # (poly_idx, vtx_idx)
        # 目前游標位置(widget 座標);用來畫 rubber band 預覽
        self._cursor_widget: QPointF | None = None

    # ---- public API --------------------------------------------------

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._img_size = (pixmap.width(), pixmap.height())
        self.update()

    def set_polygons(self, polys_xy: list[list[list[float]]]) -> None:
        self._polygons = [
            [QPointF(float(p[0]), float(p[1])) for p in poly]
            for poly in polys_xy
        ]
        self._drawing = None
        self._selected = -1
        self._drag = None
        self.polygons_changed.emit()
        self.selection_changed.emit(-1)
        self.drawing_state_changed.emit(False)
        self.update()

    def get_polygons(self) -> list[list[list[float]]]:
        return [
            [[p.x(), p.y()] for p in poly]
            for poly in self._polygons
        ]

    def start_drawing(self) -> None:
        self._finish_drag()
        self._drawing = []
        self._selected = -1
        self.drawing_state_changed.emit(True)
        self.selection_changed.emit(-1)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_drawing(self) -> None:
        if self._drawing is None:
            return
        self._drawing = None
        self.drawing_state_changed.emit(False)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def delete_selected_polygon(self) -> None:
        if 0 <= self._selected < len(self._polygons):
            del self._polygons[self._selected]
            self._selected = -1
            self.polygons_changed.emit()
            self.selection_changed.emit(-1)
            self.update()

    def set_selected(self, idx: int) -> None:
        if -1 <= idx < len(self._polygons):
            self._selected = idx
            self.selection_changed.emit(idx)
            self.update()

    def selected(self) -> int:
        return self._selected

    # ---- paint --------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0a0c12"))

        if self._pixmap is None:
            self._draw_placeholder(p)
            p.end()
            return

        # 顯示 frame(scale to fit,保留 aspect ratio)
        scale, ox, oy, sw, sh = self._scale_offset()
        p.drawPixmap(
            int(ox), int(oy), int(sw), int(sh),
            self._pixmap,
        )

        # 已完成多邊形
        for i, poly in enumerate(self._polygons):
            self._paint_polygon(p, poly, closed=True, selected=(i == self._selected))

        # 繪製中多邊形 + rubber band + snap halo
        if self._drawing is not None:
            self._paint_drawing(p)
            self._paint_status_overlay(p)

        p.end()

    def _draw_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor("#6b7280"))
        f = QFont()
        f.setPointSize(14)
        p.setFont(f)
        p.drawText(
            self.rect(), Qt.AlignmentFlag.AlignCenter,
            "正在載入影片第一幀…",
        )

    def _paint_polygon(
        self,
        p: QPainter,
        poly_img: list[QPointF],
        *,
        closed: bool,
        selected: bool,
    ) -> None:
        """畫已收尾的多邊形(idle 清單裡的)。"""
        if not poly_img:
            return
        pts = [self._image_to_widget(pt) for pt in poly_img]

        # 填色
        if len(pts) >= 3:
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            path.closeSubpath()
            fill = QColor("#f5a524")
            fill.setAlpha(55 if not selected else 85)
            p.setBrush(fill)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

        # 邊線
        pen = QPen(QColor("#f5a524" if selected else "#ffd374"))
        pen.setWidthF(2.5 if selected else 1.8)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if len(pts) >= 3:
            p.drawPolygon(QPolygonF(pts))
        else:
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

        # 頂點
        p.setPen(QPen(QColor("#1a1200"), 1.5))
        for pt in pts:
            col = QColor("#f5a524" if selected else "#ffd374")
            p.setBrush(col)
            p.drawEllipse(pt, _VERTEX_RADIUS, _VERTEX_RADIUS)

    def _paint_drawing(self, p: QPainter) -> None:
        """繪製中的多邊形:已放點 + 到游標的 rubber band + 首頂點 snap halo。"""
        assert self._drawing is not None
        pts = [self._image_to_widget(pt) for pt in self._drawing]

        can_close = len(pts) >= 3
        snap_active = can_close and self._is_snapping_to_first()

        # 已放的邊(實線,到第二點起)
        if len(pts) >= 2:
            solid = QPen(QColor("#f5a524"))
            solid.setWidthF(2.2)
            p.setPen(solid)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

        # Rubber band:從最後一個點連到游標(dashed)
        if pts and self._cursor_widget is not None:
            rb = QPen(QColor("#ffd374"))
            rb.setWidthF(1.4)
            rb.setStyle(Qt.PenStyle.DashLine)
            p.setPen(rb)
            p.drawLine(pts[-1], self._cursor_widget)

            # 游標靠近首頂點時,再畫一條 rubber band 預覽收尾
            if snap_active:
                p.drawLine(self._cursor_widget, pts[0])

        # 繪製中 snap 收尾預覽填色
        if snap_active and len(pts) >= 3:
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            path.closeSubpath()
            fill = QColor("#f5a524")
            fill.setAlpha(35)
            p.setBrush(fill)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

        # 頂點 — drawing 時更大
        p.setPen(QPen(QColor("#1a1200"), 1.5))
        p.setBrush(QColor("#f5a524"))
        for i, pt in enumerate(pts):
            r = _VERTEX_RADIUS_DRAWING if i == 0 else _VERTEX_RADIUS
            p.drawEllipse(pt, r, r)

        # 首頂點 snap halo:游標近時畫一圈 + 中心加粗
        if snap_active and pts:
            halo = QPen(QColor("#ffd374"))
            halo.setWidthF(2.0)
            p.setPen(halo)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pts[0], _SNAP_RADIUS, _SNAP_RADIUS)
            p.setBrush(QColor("#ffffff"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pts[0], _VERTEX_RADIUS_DRAWING - 2, _VERTEX_RADIUS_DRAWING - 2)

    def _paint_status_overlay(self, p: QPainter) -> None:
        """繪製中於左上角顯示狀態膠囊。"""
        assert self._drawing is not None
        n = len(self._drawing)
        if n == 0:
            text = "繪製中  ·  點畫布放第 1 個頂點  ·  Esc 取消"
        elif n < 3:
            text = f"繪製中  ·  已放 {n} 個頂點  ·  還需要 {3 - n} 個以上才能收尾"
        else:
            text = f"繪製中  ·  {n} 個頂點  ·  點首頂點 / Enter / 雙擊 收尾"

        f = QFont()
        f.setPointSize(10)
        f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        fm = p.fontMetrics()
        pad_x, pad_y = 12, 6
        tw = fm.horizontalAdvance(text) + pad_x * 2
        th = fm.height() + pad_y * 2
        rect = QRectF(12, 12, tw, th)

        p.setBrush(QColor(10, 12, 18, 220))
        p.setPen(QPen(QColor("#f5a524"), 1.2))
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QColor("#ffd374"))
        p.drawText(
            rect.adjusted(pad_x, pad_y, -pad_x, -pad_y),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _is_snapping_to_first(self) -> bool:
        """繪製中游標是否進入首頂點 snap 範圍。"""
        if (
            self._drawing is None
            or len(self._drawing) < 3
            or self._cursor_widget is None
        ):
            return False
        first_widget = self._image_to_widget(self._drawing[0])
        dx = first_widget.x() - self._cursor_widget.x()
        dy = first_widget.y() - self._cursor_widget.y()
        return (dx * dx + dy * dy) ** 0.5 <= _SNAP_RADIUS

    # ---- mouse --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        widget_pt = event.position()

        if event.button() == Qt.MouseButton.RightButton:
            # 先 check 有沒有中頂點 → context menu
            hit = self._hit_vertex(widget_pt)
            if hit is not None:
                self._show_vertex_menu(event.globalPosition().toPoint(), hit)
                return
            poly_idx = self._hit_polygon_body(widget_pt)
            if poly_idx is not None:
                self._show_polygon_menu(event.globalPosition().toPoint(), poly_idx)
                return
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        # 繪製中:先看有沒有 snap 到首頂點(即使游標在影像外也可 snap)
        if self._drawing is not None and self._is_snapping_to_first():
            self._commit_drawing()
            return

        image_pt = self._widget_to_image(widget_pt)
        # 繪製中:click 超出影像邊界 → clamp 到最近邊界點,避免「點了沒反應」
        if self._drawing is not None and not self._point_inside_image(image_pt):
            image_pt = self._clamp_to_image(image_pt)
            self._drawing.append(image_pt)
            self.update()
            return

        if not self._point_inside_image(image_pt):
            return

        # 繪製中 → 點擊加頂點
        if self._drawing is not None:
            self._drawing.append(image_pt)
            self.update()
            return

        # Idle 狀態:
        # 1) 若點到頂點 → 開始拖
        hit = self._hit_vertex(widget_pt)
        if hit is not None:
            self._drag = hit
            self._selected = hit[0]
            self.selection_changed.emit(self._selected)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            return

        # 2) 若點到多邊形 body → 選取
        poly_idx = self._hit_polygon_body(widget_pt)
        if poly_idx is not None:
            self._selected = poly_idx
            self.selection_changed.emit(poly_idx)
            self.update()
            return

        # 3) 空白處 → 取消選取
        if self._selected != -1:
            self._selected = -1
            self.selection_changed.emit(-1)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # 記下游標給 rubber band / snap halo 用
        self._cursor_widget = event.position()

        if self._drag is not None and self._pixmap is not None:
            img_pt = self._widget_to_image(event.position())
            img_pt = self._clamp_to_image(img_pt)
            pi, vi = self._drag
            if 0 <= pi < len(self._polygons) and 0 <= vi < len(self._polygons[pi]):
                self._polygons[pi][vi] = img_pt
                self.update()
            return

        # Hover cursor hint
        hit = self._hit_vertex(event.position())
        if hit is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._drawing is not None:
            # 繪製中 snap 到首頂點時 → pointing cursor 暗示可收尾
            if self._is_snapping_to_first():
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()  # repaint rubber band
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_widget = None
        if self._drawing is not None:
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._finish_drag()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drawing is not None:
            self._commit_drawing()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self._drawing is not None:
                self.cancel_drawing()
                return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._drawing is not None:
                self._commit_drawing()
                return
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_polygon()
            return
        super().keyPressEvent(event)

    # ---- helpers ------------------------------------------------------

    def _finish_drag(self) -> None:
        if self._drag is not None:
            self._drag = None
            self.polygons_changed.emit()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _commit_drawing(self) -> None:
        if self._drawing is None or len(self._drawing) < 3:
            # 少於 3 點不收尾,也 不取消;使用者可能還要繼續
            return
        self._polygons.append(self._drawing)
        self._drawing = None
        self._selected = len(self._polygons) - 1
        self.polygons_changed.emit()
        self.selection_changed.emit(self._selected)
        self.drawing_state_changed.emit(False)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _scale_offset(self) -> tuple[float, float, float, float, float]:
        if self._pixmap is None:
            return (1.0, 0.0, 0.0, 0.0, 0.0)
        iw, ih = self._img_size
        ww, wh = self.width(), self.height()
        scale = min(ww / iw, wh / ih)
        sw = iw * scale
        sh = ih * scale
        ox = (ww - sw) / 2
        oy = (wh - sh) / 2
        return (scale, ox, oy, sw, sh)

    def _widget_to_image(self, widget_pt: QPointF) -> QPointF:
        scale, ox, oy, _, _ = self._scale_offset()
        if scale <= 0:
            return QPointF(0, 0)
        return QPointF((widget_pt.x() - ox) / scale, (widget_pt.y() - oy) / scale)

    def _image_to_widget(self, img_pt: QPointF) -> QPointF:
        scale, ox, oy, _, _ = self._scale_offset()
        return QPointF(img_pt.x() * scale + ox, img_pt.y() * scale + oy)

    def _point_inside_image(self, img_pt: QPointF) -> bool:
        iw, ih = self._img_size
        return 0 <= img_pt.x() <= iw and 0 <= img_pt.y() <= ih

    def _clamp_to_image(self, img_pt: QPointF) -> QPointF:
        iw, ih = self._img_size
        return QPointF(
            max(0.0, min(iw, img_pt.x())),
            max(0.0, min(ih, img_pt.y())),
        )

    def _hit_vertex(self, widget_pt: QPointF) -> tuple[int, int] | None:
        for pi, poly in enumerate(self._polygons):
            for vi, v in enumerate(poly):
                w = self._image_to_widget(v)
                if (w - widget_pt).manhattanLength() <= _HIT_RADIUS * 1.3:
                    dx = w.x() - widget_pt.x()
                    dy = w.y() - widget_pt.y()
                    if (dx * dx + dy * dy) ** 0.5 <= _HIT_RADIUS:
                        return (pi, vi)
        return None

    def _hit_polygon_body(self, widget_pt: QPointF) -> int | None:
        img_pt = self._widget_to_image(widget_pt)
        for pi, poly in enumerate(self._polygons):
            if len(poly) < 3:
                continue
            if QPolygonF(poly).containsPoint(img_pt, Qt.FillRule.OddEvenFill):
                return pi
        return None

    # ---- context menus ------------------------------------------------

    def _show_vertex_menu(self, global_pos, hit: tuple[int, int]) -> None:  # noqa: ANN001
        pi, vi = hit
        menu = QMenu(self)
        del_act = menu.addAction("刪除此頂點")
        act = menu.exec(global_pos)
        if act is del_act:
            if 0 <= pi < len(self._polygons):
                if len(self._polygons[pi]) <= 3:
                    QMessageBox.warning(
                        self, "無法刪除",
                        "多邊形至少需要 3 個頂點。若要移除請刪除整個多邊形。",
                    )
                    return
                del self._polygons[pi][vi]
                self.polygons_changed.emit()
                self.update()

    def _show_polygon_menu(self, global_pos, pi: int) -> None:  # noqa: ANN001
        menu = QMenu(self)
        del_act = menu.addAction("刪除整個多邊形")
        act = menu.exec(global_pos)
        if act is del_act:
            if 0 <= pi < len(self._polygons):
                del self._polygons[pi]
                self._selected = -1
                self.polygons_changed.emit()
                self.selection_changed.emit(-1)
                self.update()


class RoiEditorView(QWidget):
    """違停檢測專屬的 ROI 編輯頁。"""

    rois_saved = Signal(str)      # project_path
    cancelled = Signal()          # 使用者按返回 → 關專案

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_path: Path | None = None
        self._video_path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._fps: float = 30.0

        self._build_ui()

    # ---- layout ------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # 頂部 header
        header = QHBoxLayout()
        header.setSpacing(10)
        title = QLabel("違停檢測 · 畫違法區")
        tf = QFont()
        tf.setPointSize(15)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)
        self.hint_lbl = QLabel(
            "左鍵加頂點 / 點首頂點或雙擊 / Enter 收尾 / 拖頂點移動 / 右鍵刪頂點 / Esc 取消"
        )
        self.hint_lbl.setObjectName("Hint")

        header.addWidget(title)
        header.addSpacing(12)
        header.addWidget(self.hint_lbl)
        header.addStretch(1)

        self.back_btn = QPushButton("← 返回")
        self.back_btn.setProperty("ghost", True)
        self.back_btn.clicked.connect(self._on_back)

        self.skip_btn = QPushButton("略過(不畫 ROI)")
        self.skip_btn.setProperty("ghost", True)
        self.skip_btn.clicked.connect(self._on_skip)

        self.save_btn = QPushButton("儲存並繼續 →")
        self.save_btn.setProperty("accent", True)
        self.save_btn.setMinimumWidth(150)
        self.save_btn.clicked.connect(self._on_save)

        header.addWidget(self.back_btn)
        header.addWidget(self.skip_btn)
        header.addWidget(self.save_btn)
        root.addLayout(header)

        # 進階設定列(停留門檻 + 位移容忍)
        adv_row = QHBoxLayout()
        adv_row.setSpacing(10)
        adv_lbl = QLabel("進階")
        adv_lbl.setObjectName("MetricLabel")

        self.stopped_sec_spin = QSpinBox()
        self.stopped_sec_spin.setRange(1, 3600)
        self.stopped_sec_spin.setValue(60)
        self.stopped_sec_spin.setSuffix("  秒")
        self.stopped_sec_spin.setFixedWidth(100)
        self.stopped_sec_spin.setToolTip(
            "車輛靜止超過此秒數才列為候選。台北常見紅燈 30-50s,預設 60s "
            "可濾掉多數車陣 / 紅燈;短影片測試可改成 2-5s。"
        )

        self.stopped_disp_spin = QDoubleSpinBox()
        self.stopped_disp_spin.setRange(0.0, 200.0)
        self.stopped_disp_spin.setValue(20.0)
        self.stopped_disp_spin.setDecimals(0)
        self.stopped_disp_spin.setSuffix("  px")
        self.stopped_disp_spin.setFixedWidth(100)
        self.stopped_disp_spin.setToolTip(
            "車輛 bbox 底邊中心相對於段落起點位移超過此像素數即視為在移動。"
        )

        adv_hint = QLabel(
            "停留門檻越小,短暫停車也會被抓(例如兩三秒測試影片設 2-3 秒)。"
        )
        adv_hint.setObjectName("Hint")

        adv_row.addWidget(adv_lbl)
        adv_row.addWidget(QLabel("停留門檻"))
        adv_row.addWidget(self.stopped_sec_spin)
        adv_row.addSpacing(10)
        adv_row.addWidget(QLabel("位移容忍"))
        adv_row.addWidget(self.stopped_disp_spin)
        adv_row.addSpacing(14)
        adv_row.addWidget(adv_hint, stretch=1)
        root.addLayout(adv_row)

        # 主區:左側 polygon 清單 + 中間 canvas
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # 左側
        left = QFrame()
        left.setObjectName("Card")
        left.setProperty("class", "card")
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(8)
        lbl = QLabel("違法區")
        lbl.setObjectName("MetricLabel")
        self.polygon_list = QListWidget()
        self.polygon_list.setStyleSheet(
            "QListWidget { background: transparent; border: none; padding: 0; }"
            "QListWidget::item { padding: 6px 8px; border-radius: 4px; margin-bottom: 2px; }"
            "QListWidget::item:selected { background: rgba(245,165,36,0.18); color: #f5a524; }"
        )
        self.polygon_list.currentRowChanged.connect(self._on_list_row_changed)

        self.new_btn = QPushButton("+ 新增多邊形")
        self.new_btn.clicked.connect(self._on_new_polygon)
        self.delete_btn = QPushButton("刪除選取")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._on_delete_polygon)

        ll.addWidget(lbl)
        ll.addWidget(self.polygon_list, stretch=1)
        ll.addWidget(self.new_btn)
        ll.addWidget(self.delete_btn)

        # 中間:畫布
        self.canvas = _RoiCanvas()
        self.canvas.polygons_changed.connect(self._refresh_list)
        self.canvas.selection_changed.connect(self._on_canvas_selection)
        self.canvas.drawing_state_changed.connect(self._on_drawing_state)

        main_row.addWidget(left)
        main_row.addWidget(self.canvas, stretch=1)
        root.addLayout(main_row, stretch=1)

        # 底部:frame slider
        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)
        slider_row.addWidget(QLabel("背景幀:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_time_lbl = QLabel("00:00.00")
        self.frame_time_lbl.setObjectName("Hint")
        self.frame_time_lbl.setMinimumWidth(80)
        slider_row.addWidget(self.frame_slider, stretch=1)
        slider_row.addWidget(self.frame_time_lbl)
        root.addLayout(slider_row)

    # ---- public ------------------------------------------------------

    def load_project(self, project_path: Path) -> None:
        self._project_path = project_path
        project = Project.load(project_path)
        try:
            self._video_path = Path(project.meta.video_path)
            existing = list(project.meta.no_parking_zones or [])
            cur_stopped_sec = float(project.meta.stopped_threshold_sec or 60.0)
            cur_stopped_disp = float(project.meta.stopped_max_displacement_px or 20.0)
            self._fps = project.meta.video_fps or 30.0
            self._total_frames = project.meta.video_frame_count or 0
        finally:
            project.close()

        if not self._video_path.is_file():
            QMessageBox.warning(
                self, "找不到影片",
                f"專案記錄的影片不存在:\n{self._video_path}",
            )
            return

        self._open_video()
        self.frame_slider.setRange(0, max(0, self._total_frames - 1))
        self.frame_slider.setValue(0)
        self._load_frame(0)
        self.canvas.set_polygons(existing)
        self._refresh_list()
        # 套用已存的進階設定閾值(新專案 = 預設)
        self.stopped_sec_spin.setValue(int(round(cur_stopped_sec)))
        self.stopped_disp_spin.setValue(float(cur_stopped_disp))
        self.canvas.setFocus()

    def shutdown(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---- video helpers ----------------------------------------------

    def _open_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._video_path is None:
            return
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            QMessageBox.critical(
                self, "無法開啟影片", f"OpenCV 無法開啟:{self._video_path}"
            )
            return
        self._cap = cap
        if self._total_frames <= 0:
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self._fps <= 0:
            self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def _load_frame(self, frame_idx: int) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(QPixmap.fromImage(img))
        t_sec = frame_idx / self._fps if self._fps > 0 else 0.0
        m = int(t_sec) // 60
        s = t_sec - 60 * m
        self.frame_time_lbl.setText(f"{m:02d}:{s:05.2f}")

    # ---- slots -------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        self._load_frame(value)

    def _on_new_polygon(self) -> None:
        self.canvas.start_drawing()
        self.canvas.setFocus()

    def _on_delete_polygon(self) -> None:
        self.canvas.delete_selected_polygon()

    def _on_list_row_changed(self, row: int) -> None:
        self.canvas.set_selected(row)

    def _on_canvas_selection(self, idx: int) -> None:
        self.polygon_list.blockSignals(True)
        self.polygon_list.setCurrentRow(idx)
        self.polygon_list.blockSignals(False)
        self.delete_btn.setEnabled(idx >= 0)

    def _on_drawing_state(self, drawing: bool) -> None:
        self.new_btn.setEnabled(not drawing)
        if drawing:
            self.hint_lbl.setText(
                "繪製中 — 左鍵放頂點 · 點首頂點 / Enter / 雙擊 收尾 · Esc 取消"
            )
        else:
            self.hint_lbl.setText(
                "左鍵加頂點 / 點首頂點或雙擊 / Enter 收尾 / 拖頂點移動 / 右鍵刪頂點 / Esc 取消"
            )

    def _refresh_list(self) -> None:
        polys = self.canvas.get_polygons()
        self.polygon_list.blockSignals(True)
        cur = self.polygon_list.currentRow()
        self.polygon_list.clear()
        for i, poly in enumerate(polys):
            item = QListWidgetItem(f"違法區 {chr(ord('A') + i)}  ({len(poly)} 頂點)")
            self.polygon_list.addItem(item)
        if 0 <= cur < len(polys):
            self.polygon_list.setCurrentRow(cur)
        self.polygon_list.blockSignals(False)

    def _on_back(self) -> None:
        polys = self.canvas.get_polygons()
        if polys:
            r = QMessageBox.question(
                self, "返回?",
                "目前已畫的多邊形將不會儲存。確定返回嗎?",
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        self.cancelled.emit()

    def _on_skip(self) -> None:
        r = QMessageBox.question(
            self, "略過 ROI?",
            "不畫違法區的話,所有停著的車都會成為候選(包含合法停車位的)。\n"
            "確定略過嗎?",
        )
        if r != QMessageBox.StandardButton.Yes:
            return
        self._save_and_emit([])

    def _on_save(self) -> None:
        polys = self.canvas.get_polygons()
        if not polys:
            r = QMessageBox.question(
                self, "沒有多邊形?",
                "還沒畫任何違法區。是否直接儲存(= 略過)?",
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        self._save_and_emit(polys)

    def _save_and_emit(self, polys: list[list[list[float]]]) -> None:
        if self._project_path is None:
            return
        try:
            project = Project.load(self._project_path)
            try:
                project.save_no_parking_zones(polys)
                project.save_static_thresholds(
                    stopped_threshold_sec=self.stopped_sec_spin.value(),
                    stopped_max_displacement_px=self.stopped_disp_spin.value(),
                )
                project.update_progress("roi_saved", zones=len(polys))
            finally:
                project.close()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "儲存失敗", f"無法儲存 ROI:\n{exc}")
            return
        self.rois_saved.emit(str(self._project_path))
