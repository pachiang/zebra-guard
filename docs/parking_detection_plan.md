# 違停檢測藍圖(靜態攝影機)

> 與未禮讓行人檢測獨立的**第二條功能線**。共用 UI 殼 / Project / export 基礎設施,但使用獨立的 ROI 編輯器、規則引擎、Review 標籤系統。接手前先讀 `docs/pipelines.md` 了解整體模式 × backend 架構;本文件只涵蓋 Static mode 下「違停檢測」這一條。

## 1. 定位

| 項目 | 內容 |
|---|---|
| 情境 | 路邊固定式攝影機,影片長度從幾分鐘到數小時 |
| 相機假設 | **不動**(影片中途若相機被搬動,ROI 就失效,需要重畫) |
| 使用者輸入 | 一個或多個「合法停車區」polygon + 影片 |
| 輸出 | 候選違停事件清單,使用者標籤後匯出 |
| 不做 | 自動分類「並排 / 路口 / 其他」、lane / curb 自動偵測、homography |

## 2. 使用者流程

```
[Home 三卡] → [違停檢測 · 新建]
   → [選影片] (mp4 / mov / mkv)
   → [ROI 編輯器]      ← 使用者畫合法停車區多邊形(可畫多個)
   → [設定門檻]        ← 可選:停留時間、位移容忍
   → [Processing]      ← YOLO + ByteTrack(無 Mask2Former / YOLO-seg)
   → [Review]          ← 看候選 + 下拉標籤 + 採用 / 拒絕
   → [匯出]            ← 只匯出標記為違停類別的事件
```

## 3. App 入口改動

`ImportView` 改為首頁三卡:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ 未禮讓行人       │  │ 違停檢測         │  │ 開啟既有專案     │
│ (dashcam)       │  │ (靜態攝影機)     │  │                  │
│ 行車記錄器影片    │  │ 固定相機影片     │  │                  │
│ [新建 →]         │  │ [新建 →]         │  │ [選擇…]          │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

- 新專案 wizard 依點擊卡片決定 `Project.mode = dashcam | static`
- 開啟既有:根據 `project.meta.mode` 路由到對應 Review 介面

## 4. ROI 編輯器(合法停車區)

### 4.1 元件
`ui/roi_editor_view.py`(新)
- 左側工具:第一幀 / 指定幀(slider)選擇背景影像
- 中央:QGraphicsView 承載多邊形
- 互動:
  - 左鍵加頂點;double-click 收尾成 polygon
  - 拖曳頂點移動;右鍵頂點刪除
  - 已畫好的 polygon 可選中、整體拖移、整體刪除
- 右側:已畫 polygon 清單(顯示點數、重新命名「合法停車區 A / B…」)

### 4.2 Bottom-strip 判定(核心設計)

**不用整個 bbox 判斷**是否在 ROI 內,因為:
- 遠處車輛 bbox 會偏大,車頂可能跨入 ROI 但輪子(真正落地處)在 ROI 外
- 透視扭曲讓「看似在 ROI 裡」不等於「物理在合法停車區」

改用 bbox 的**下 20% 水平條帶**(bottom strip):
```
area(strip ∩ roi_union) / area(strip) >= 0.5   → 視為在 ROI 內(合法)
                                          < 0.5  → 視為在 ROI 外(候選違停)
```

- 多個 ROI 先 union 再計算(使用者可能畫多個合法停車區)
- 0.5 門檻可調,實測後決定
- 此邏輯與 v7 crosswalk rule 的 `_bbox_component` 同源

### 4.3 資料
`ProjectMeta` 新欄位(static mode 專用):
```python
parking_zones: list[list[list[float]]]   # 多個 polygon,每個是 [[x,y], ...]
```

## 5. 「停著」判定

### 5.1 單車判定
同一 `track_id` 在連續時段內:
- 長度 ≥ `stopped_threshold_sec`
- bbox 中心位移 ≤ `stopped_max_displacement_px`

**預設 `stopped_threshold_sec = 60`**(不是 15),理由見 § 5.2。

### 5.2 紅燈 / 車陣處理

紅燈、車陣、暫停讓行人都會產生「看起來停著」的車。**兩層設計**:

**A. 時間門檻(主要防線,第一版做)**
- 預設 `stopped_threshold_sec = 60` — 台北常見紅燈 30-50 秒,60 秒閾值濾掉九成以上紅燈
- 車陣通常在 60 秒內至少滾動幾公尺(中心位移 > 20 px 就不算「停著」)
- 真停車動輒數分鐘,絕對 > 60s
- **使用者可透過「進階設定」dialog 調整秒數**(沿用既有 `AdvancedSettingsDialog` 架構,加 static mode 專屬欄位組:`stopped_threshold_sec` / `stopped_max_displacement_px`)
- 60s 太嚴 → 下調至 30-45s;誤判太多 → 上調至 90-120s

**B. 使用者標籤兜底(Review 永遠有)**
- A 漏網的紅燈 / 車陣候選,Review 頁下拉選 `red_light` 標籤即可
- 匯出時自動排除
- 本項是**設計上的 safety net**,不是「要做的事」,是資料流本來就有

**暫緩(第一版不做)**:同時停止聚類(detect ≥ 4 台同時靜止 → 自動標為車陣)。實作負擔大、行為容易讓使用者困惑、且 A + B 已夠用。觀察 MVP 實際誤判量再評估。

## 6. 候選事件產生(不做自動分類)

對每個 vehicle_track:
1. 找出所有「停著」時段(§ 5)
2. 對每個停著時段,檢查 bbox bottom-strip 是否在任一 ROI 外(§ 4.2)
3. 若是 → 產生一筆 `ParkingEvent`
 - `start_frame / end_frame`
 - `start_sec / end_sec`
 - `vehicle_track_id`
 - `vehicle_class`(car / bus / truck;motorcycle 第一版**不收**,規則不同)
 - `_display_thumbnail`(事件中段一幀)
4. 無任何「並排」、「路口」等分類判定

## 7. 使用者標籤系統(取代自動分類)

Review 頁每個候選附下拉選單:

| 標籤 key | 中文 | 匯出? |
|---|---|---|
| `parallel_park` | 並排違停 | ✓ |
| `intersection` | 路口違停 | ✓ |
| `other_violation` | 其他違停(紅黃線、人行道、禁停標誌) | ✓ |
| `red_light` | 等紅燈 / 車陣 | ✗ |
| `legal_elsewhere` | 合法(使用者漏畫 ROI) | ✗ |
| `ignore` | 忽略(誤偵測、非車、畫面外) | ✗ |

使用者也可單選 accept / reject(保留原流程)。**匯出條件**:`user_status == 'accepted'` AND `user_label ∈ {parallel_park, intersection, other_violation}`。

### 快捷鍵(複用既有)
- `A` 採用、`R` 拒絕、`J/K` 上下
- `1-6` 快速選標籤(對應上表六類)
- `Del` 刪除

## 8. 資料模型變動

### 8.1 `ProjectMeta`
```python
mode: str = "dashcam"   # 現有:"dashcam" | "static"
parking_zones: list[list[list[float]]] = []     # 新:每個 polygon 一個內層 list
stopped_threshold_sec: float = 60.0             # 新
stopped_max_displacement_px: float = 20.0       # 新
parking_vehicle_classes: list[int] = [2, 5, 7]  # 新:car / bus / truck COCO ids
```

### 8.2 `events` 表
現有欄位:`id, start_frame, end_frame, start_sec, end_sec, ped_track_ids, veh_track_ids, user_status, user_note, license_plate`。

新增欄位(migration 同車牌,用 `PRAGMA table_info` 偵測缺欄再 ALTER):
```sql
user_label   TEXT        -- 見 § 7 的 6 類 key;NULL = 未標
```

`min_distance_px` / `peak_speed_px` 在 static mode 不適用 → 存 0 / 0 或 NULL。`ped_track_ids` 永遠空。`veh_track_ids` 只存一個 track id(就是那台停著的車)。

## 9. ML / 規則模組(新)

`src/zebraguard/ml/parking_rules.py`(新):

```python
from dataclasses import dataclass

@dataclass
class StoppedTrack:
    track_id: int
    start_frame: int
    end_frame: int
    representative_bbox: tuple[float, float, float, float]  # 中段幀的 bbox


def find_stopped_tracks(
    tracks: list[Track],
    min_sec: float,
    max_disp_px: float,
    fps: float,
) -> list[StoppedTrack]:
    """對每個 track 找出「中心位移 <= max_disp_px 且持續 >= min_sec」的時段。
    一個 track 可能產生多筆(先停、再動、再停)。"""
    ...


def bottom_strip_outside_roi(
    bbox: tuple[float, float, float, float],
    roi_polygons: list[np.ndarray],
    threshold: float = 0.5,
    strip_frac: float = 0.20,
) -> bool:
    """bbox 底部 strip 與 ROI union 的覆蓋率 < threshold → 回傳 True(在 ROI 外)。"""
    ...


def build_parking_candidates(
    tracks: list[Track],
    parking_zones: list[list[tuple[float, float]]],
    cfg: ParkingConfig,
    fps: float,
) -> list[ParkingEvent]:
    """完整候選產生流程。"""
    ...
```

**不需要**:Mask2Former、YOLO-seg、crosswalk source、homography。**需要**:既有 `ml/detection.py` 的 `track_video()`(共用 YOLO + ByteTrack)。

## 10. Pipeline 編排

`scripts/zero_shot_detect.py` 的 mask2former / yolo_seg 跟 static mode 關係小,改法兩種:

**A. 另建 `scripts/static_parking_detect.py`**(推薦)
- 完全獨立的 CLI,共用 `ml/detection.py` 的 YOLO tracking
- 較短,易維護,不污染 dashcam 路徑
- Worker 依 `project.meta.mode` 選擇呼叫哪個 script

**B. 把 static 塞進 `zero_shot_detect.py`**
- 省一個檔案但變成 God object
- 不建議

## 11. Worker 分流

`core/worker.py` 現有 `_BACKEND_PRESET` 只管 dashcam 的 backend;static mode 需要另一條路徑:

```python
def _run_inner(self):
    ...
    if mode == "dashcam":
        # 現行:load zero_shot_detect.py + run(crosswalk_backend=...)
        ...
    elif mode == "static":
        # 新:load static_parking_detect.py + run(roi_polygons=..., stopped_threshold=...)
        ...
```

## 12. UI views(Static mode)

| View | 用途 | 複用 / 新增 |
|---|---|---|
| `ImportView` | 三卡首頁 + 既有開啟 | 擴 |
| `StaticProjectWizard`(新) | 確認影片 + 選 ROI 編輯 | 新(類似現行 `NewProjectDialog` 精簡版) |
| `RoiEditorView`(新) | 畫合法停車區 polygon | 新 |
| `ParkingSettingsDialog`(可選) | 進階門檻 | 可沿用 `AdvancedSettingsDialog` 的外殼 |
| `ProcessingView` | 進度條 | 完全複用 |
| `ParkingReviewView`(新) | 候選清單 + 標籤下拉 | 從 `ReviewView` 抽 base |
| `ExportDialog` | 匯出 | 複用,但匯出過濾改看 `user_label` |

**ReviewView 是否重構**:抽 base class `BaseReviewView`,dashcam 與 parking 各自繼承,各自擴充標籤系統。MVP 階段可以先 copy-paste,之後再抽。

## 13. 匯出與檢舉書

匯出流程複用既有 `export/` 的 clip / keyframe / complaint draft,但:
- 過濾規則:`user_status == 'accepted' AND user_label in {parallel_park, intersection, other_violation}`
- 檢舉書模板換違停用(`道路交通管理處罰條例 第 55、56 條`)
- 車牌欄位複用;時間欄位複用;違規類型從 `user_label` 展開成中文文字

建議 `export/complaint.py` 改支援多個模板,以 `mode` + `user_label` 選擇對應文字。

## 14. 實作 Milestones(按順序)

| # | 內容 | 預估 |
|---|---|---|
| M1 | `ImportView` 三卡 + Project `mode=static` 路由 | 0.5 day |
| M2 | `RoiEditorView`(QGraphicsView 多邊形編輯) | 1-2 day |
| M3 | `ml/parking_rules.py` + 單元測試 | 0.5 day |
| M4 | `scripts/static_parking_detect.py` + worker 分流 | 1 day |
| M5 | `ParkingReviewView`(含標籤下拉、1-6 快捷鍵) | 1 day |
| M6 | 匯出過濾 + 違停檢舉書模板 | 0.5 day |
| M6.5 | `AdvancedSettingsDialog` 新增 static mode 欄位組(`stopped_threshold_sec` / `stopped_max_displacement_px` / bottom-strip 閾值) | 0.5 day |
| M7 | `docs/mvp.md` / `docs/pipelines.md` 回填 + user-guide 一頁 | 0.5 day |

**總計 ~5-5.5 天**。比 YOLO-seg backend 整合大一些,但完全不用碰 ML 模型訓練。

## 15. 已決定(避免重複討論)

- Motorcycle / scooter **第一版不收** — 規則不同(機車格常畫在人行道、紅線),加入只會多噪音。Phase 2 再評估。
- **不做自動並排 / 路口分類**。全部候選列出,使用者標。
- **不做** lane / curb auto-detection。使用者手畫 ROI。
- Static mode **不支援** dashcam 樣影片。影片中途相機被移位,ROI 失效,需重新建立專案。
- `stopped_threshold_sec` **預設 60 秒**;使用者可在進階設定 dialog 調整。
- 紅燈 / 車陣處理 = **時間門檻(A)+ 使用者標籤兜底(C)**。聚類啟發式(B)先不做。
- **ReviewView 先 copy-paste**,不強迫先抽 base class;寫出兩個實作後再重構。

## 16. 未決 TODO

- [ ] M2 的 RoiEditorView 要不要用 `QGraphicsView` / 自繪 / 第三方?(目前規劃 QGraphicsView,成熟度最高)
- [ ] `ParkingConfig` 裡是否加「紅黃線自動偵測」?會不會有用?第一版不做,看實際誤判量
- [ ] `user_label` enum 是否要可自訂?例如使用者想加「貨車卸貨」分類。第一版固定 6 類
- [ ] 單一 track 若「停 → 動幾公尺 → 又停」算一筆還是兩筆事件?目前規劃兩筆,Phase 2 可加合併規則
- [ ] 與警政署檢舉系統的違停欄位對應(專案結束前務必對齊)

## 17. Session-handoff 指引

接手前:
1. 讀本檔 + `docs/pipelines.md` § 「靜態攝影機 / 線路 A」
2. 確認 `Project.mode == "static"` 的 schema 與路由(現行 main 已有 mode 欄位但未使用)
3. 從 M1 開始逐一打勾;每個 milestone 單獨 commit
4. 從 dashcam 的 `core/worker.py` 的寫法學 QThread 模式,不要 copy-paste
5. Review UI 先 copy-paste,最後再抽 base(見 § 15)
