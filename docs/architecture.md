# 架構文件

> 狀態:初稿 — 跟著 MVP 實作逐步演進。標註 `TODO` 的部分是尚未決定的細節。

## 1. 設計原則

1. **全本機執行** — 不上傳任何影片,使用者可離線使用。
2. **介面層與運算層分離** — `ml/` 不可依賴 PySide6;任何 `ui/` 以外的模組都可以在 CLI 或 Jupyter 中單獨呼叫。
3. **長時任務第一等公民** — Pipeline 從第一天就假設影片 ≥ 10 分鐘:可中斷、可續跑、有 checkpoint、有誠實 ETA。
4. **審查優先於自動化** — AI 偵測結果是「候選違規」,最終判定必須經過使用者在 Review UI 確認,沒有全自動匯出。
5. **不做自動送件** — 產出檢舉輔助包,但送件動作由使用者親自完成(法律責任與 ToS 的考量,詳見 `disclaimer.md`)。
6. **零手動校準** — 斑馬線位置由 Mask2Former 以高頻(每數幀)自動偵測,使用者**不需**手繪 ROI、**不需**校正 homography。僅支援行車記錄器(dashcam)影片;固定式 CCTV 不在範圍內。

## 2. 元件圖

```
┌─────────────────────────────────────────────────────────────┐
│                        PySide6 UI                            │
│  MainWindow · ProjectWizard · ImportView                     │
│  ProcessingView · Timeline · ReviewDeck · ExportDialog       │
│                            ↓↑ Qt Signal/Slot                 │
├─────────────────────────────────────────────────────────────┤
│                    core (協調層)                              │
│  Project (.zgproj)    Pipeline (呼叫 ml.zero_shot)           │
│  Worker (QThread, cancellable, checkpoint)                   │
├─────────────────────────────────────────────────────────────┤
│    ml (純運算, 無 Qt)          export            storage       │
│  CrosswalkSeg (Mask2Former) ClipExporter      ProjectDB       │
│  Detection + Tracking       KeyframeExporter  ModelManager    │
│  (YOLO11 + ByteTrack)       ComplaintDraft                    │
│  ViolationRules (像素規則)                                     │
├─────────────────────────────────────────────────────────────┤
│    utils (GPU 偵測 · ffmpeg bin · logging · 設定)             │
└─────────────────────────────────────────────────────────────┘
```

**相依方向只能由上往下**。`ml/` 不 import `ui/`、不 import `core/`;`core/` 不 import `ui/`。這使 `ml/` 可在無 GUI 環境跑 batch、可單獨測試。

## 3. 主要資料流

1. 使用者選擇影片 → `ProjectWizard` 建立新的 `.zgproj`(純儲存影片路徑與 hash)。**沒有 ROI 編輯步驟,也沒有 homography 校正步驟**。
2. 啟動分析 → `core.Pipeline` 在 `core.Worker` (QThread) 中跑 v7 preset:
   - **每幀**(或依 `stride` 跳幀):YOLO + ByteTrack 偵測行人 / 車輛。
   - **每 N 幀**(`mask_every`,預設 5):Mask2Former 更新斑馬線 mask;mask 經膨脹後以 connected-components 區分「不同路口的不同斑馬線」。
   - **規則引擎**:行人腳點壓到 dilated mask 的 component X + 同幀上**同一** component X 有一台 bottom-strip 壓到 mask 且像素位移 > `moving_px` 的車輛,則該幀命中;連續命中經 `merge_gap_sec` 聚合為一筆事件。
3. 偵測結果即時寫入專案 SQLite,UI 透過 Qt Signal 顯示違規事件。
4. 使用者在 `ReviewDeck` 審查(採用/拒絕/調整時段);以 OpenCV 解碼該時段影片、循環播放 ±3 秒、疊加 bbox 與 mask overlay。
5. 使用者匯出 → `export.clip` 以 ffmpeg 無損剪輯 → `export.keyframe` 產關鍵幀 → `export.complaint_draft` 產檢舉書草稿。

## 4. 執行緒模型

| Thread | 職責 |
|---|---|
| UI (main) | PySide6 event loop;只處理 UI 事件、Qt Signal 接收。 |
| Worker | 跑 `ml.zero_shot_pipeline`(Mask2Former + YOLO + 規則);每數十幀以 Qt Signal 回傳進度。 |
| Player | 審查頁的影片播放;用 `cv2.VideoCapture` 於 QThread 解碼,經 Qt Signal 把每幀 QImage 丟回 UI 繪製。 |

**取消機制**:Worker 持有 `threading.Event` cancel flag,每幀頂部檢查;取消時關閉 cv2 / torch 資源後 emit `cancelled` signal。

**Checkpoint**:Worker 每 N 秒把「已處理幀數 + 累積事件」flush 到 SQLite 並更新 `project.progress_frames`。重啟時從 `progress_frames` 繼續(Mask2Former 本身無跨影片狀態,重跑幾秒鐘的 overlap 即可)。

## 5. 專案檔 (.zgproj) 格式

實作上為一個**資料夾**(未來可考慮 zip 打包),內含:

```
my_clip_2026-04-20.zgproj/
├── project.json          # 影片路徑、設定、版本、進度
├── project.db            # SQLite:每幀 hit 記錄、聚合後的事件、使用者標註
├── thumbnails/           # 違規事件關鍵幀快取
└── exports/              # 使用者匯出的片段與報告
```

`project.json` 範例結構:

```json
{
  "version": 1,
  "video": { "path": "...", "sha256_partial": "...", "duration_sec": 600, "fps": 30.0 },
  "pipeline_settings": {
    "preset": "v7_baseline",
    "stride": 3,
    "mask_every": 5,
    "dilate_px": 20,
    "moving_px": 2.0,
    "person_conf": 0.35,
    "merge_gap_sec": 0.6,
    "min_event_frames": 2,
    "exclude_ego": true,
    "mask_imgsz": 640
  },
  "progress": { "stage": "analyzing", "processed_frames": 12345, "total_frames": 18000 }
}
```

`project.db` schema 初稿(待實作時調整):

- `events(id, start_frame, end_frame, start_sec, end_sec, min_distance_px, peak_speed_px, ped_track_ids_json, veh_track_ids_json, user_status, user_note)`
- `frame_hits(frame_idx, t_sec, ped_track_ids_json, veh_track_ids_json, min_distance_px, speed_px)` — 聚合前的原始命中,供重新聚合或除錯
- `meta(key, value)`

> 與舊版(靜態 CCTV)schema 相比:已拿掉 `detections` 細表與 `tracks` 細表;改以事件導向儲存。如果日後要做逐幀檢視,可以再加回 `detections` 表。

## 6. ML 管線細節 (v7 preset)

參數與 `scripts/presets/v7_baseline_params.json` 完全一致。下列為核心步驟的程式角度說明。

### 6.1 斑馬線自動分割 (`ml/crosswalk_seg.py`)
- 模型:`facebook/mask2former-swin-large-mapillary-vistas-semantic`(Mapillary Vistas 類別包含 "Crosswalk - Plain" 與 "Lane Marking - Crosswalk")
- 執行頻率:每 `mask_every` 幀重新推論一次。因 dashcam 視角持續變化,這個頻率不能太低(預設 5)。
- `mask_imgsz`:Mask2Former 輸入時先把短邊 resize 到此像素(預設 640),輸出再放大回原解析度;可顯著降低 GPU 負載與溫度。
- 小面積 mask 過濾:若 `area < min_mask_area_frac × 影格面積`(預設 0.5%)則當幀視為無斑馬線,以避免 ghost mask。
- 膨脹後跑 connected-components,每個分量代表「同一路口的一片斑馬線」,供後續「同分量規則」使用。

### 6.2 偵測與追蹤 (`ml/detection.py`)
- YOLO11(Ultralytics 預訓練 COCO 權重)+ ByteTrack(`persist=True` 跨 stride 呼叫維持 track id)
- Execution Provider 優先序:**CUDA** → CPU(未來加 DirectML;見 § 7)
- 類別:`{person, bicycle, car, motorcycle, bus, truck}`

### 6.3 Ego vehicle / 騎士過濾 (`ml/detection.py` 內)
- **Ego vehicle**:bbox 貼著畫面底部、寬度占畫面 ≥ 40%、高度占畫面 ≥ 25% 視為自車,排除。
- **Rider**:若某 person bbox 與某 motorcycle / bicycle bbox 有足夠 containment(預設 40%),或 person 腳點落在機車 bbox 擴張 `foot_margin_px` 內,判定為騎士,不視為穿越行人。

### 6.4 違規規則 (`ml/violation_rules.py`)
同一幀內,若存在 `comp_id` X 使得:
- 行人(非騎士)bbox 的底部 strip 多數像素落在 dilated mask component X 內;且
- 車輛(非 ego)bbox 的底部 strip 多數像素也落在 component X 內;且
- 該車輛前一幀中心至本幀中心的像素位移 ≥ `moving_px`

則本幀命中 (`frame_hit`)。連續命中經 `merge_gap_sec` 聚合、小於 `min_event_frames` 的事件過濾,得到最終候選事件。

### 6.5 與舊版 homography 管線的關係
舊版(docs 早期版本、`ml/homography.py`、`ml/roi.py`、`ml/violation_rules.py` 舊實作)以公尺為單位判斷「3 公尺未禮讓」,需使用者手動校準。新版放棄此路線:
- 優點:零手動校準、程式更簡單、適用一般使用者。
- 代價:距離以像素為單位,誤判率依相機焦段而異;`legal-rules.md` 的「3 公尺」在程式中不直接對應,改由 `dilate_px` + 「同一 component」的空間重疊替代。
- `ml/homography.py` 與 `ml/roi.py` 暫時保留供 Phase 2 的固定 CCTV 支援;MVP 不在主流程中呼叫。

## 7. 模型分發

- 安裝檔綁附 `yolo11n.pt`(~6 MB)。
- Mask2Former (`mask2former-swin-large-mapillary-vistas-semantic`) 約 850 MB,**不綁附**;首次啟動由 `ModelManager` 觸發下載,預設走 HuggingFace Hub;可選擇鏡像或本地檔。
- `storage.ModelManager` 負責:下載、SHA256 校驗、版本檢查、清理。
- Hub / CDN:TODO — 先用 HuggingFace 預設;若限流再評估自架鏡像。

## 8. 設定與使用者資料位置

| 類型 | 位置 |
|---|---|
| 安裝目錄 | `%LOCALAPPDATA%\Programs\ZebraGuard\` (免管理員權限) |
| 使用者模型 | `%LOCALAPPDATA%\ZebraGuard\models\` |
| 使用者設定 | `%APPDATA%\ZebraGuard\settings.ini` (QSettings) |
| 日誌 | `%LOCALAPPDATA%\ZebraGuard\logs\` |
| 專案檔 | 使用者自選位置,預設「文件」資料夾 |

## 9. 尚未決定 (TODO)

- 依賴管理工具:`uv` vs `poetry` vs `pip-tools`(目前傾向 uv)
- 車牌 OCR 模型選型(Phase 2)
- 自動更新機制(Phase 2+)
- 國際化:預設 zh-TW,英文介面是否需要
- DirectML 支援:`onnxruntime-directml` 涵蓋 YOLO,Mask2Former 走 ONNX export 是否可行待驗證
