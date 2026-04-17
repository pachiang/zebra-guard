# 架構文件

> 狀態:初稿 — 跟著 MVP 實作逐步演進。標註 `TODO` 的部分是尚未決定的細節。

## 1. 設計原則

1. **全本機執行** — 不上傳任何影片,使用者可離線使用。
2. **介面層與運算層分離** — `ml/` 不可依賴 PySide6;任何 `ui/` 以外的模組都可以在 CLI 或 Jupyter 中單獨呼叫。
3. **長時任務第一等公民** — Pipeline 從第一天就假設影片 ≥ 1 小時:可中斷、可續跑、有 checkpoint、有誠實 ETA。
4. **審查優先於自動化** — AI 偵測結果是「候選違規」,最終判定必須經過使用者在 Review UI 確認,沒有全自動匯出。
5. **不做自動送件** — 產出檢舉輔助包,但送件動作由使用者親自完成(法律責任與 ToS 的考量,詳見 `disclaimer.md`)。

## 2. 元件圖

```
┌─────────────────────────────────────────────────────────────┐
│                        PySide6 UI                            │
│  MainWindow · ProjectWizard · RoiEditor · Calibration        │
│  ProcessingView · Timeline · ReviewDeck · ExportDialog       │
│                            ↓↑ Qt Signal/Slot                 │
├─────────────────────────────────────────────────────────────┤
│                    core (協調層)                              │
│  Project (.zgproj)    Pipeline (Stage1 → Stage2)             │
│  Worker (QThread, cancellable, checkpoint)                   │
├─────────────────────────────────────────────────────────────┤
│    ml (純運算, 無 Qt)          export            storage       │
│  Detection (ONNX)           ClipExporter      ProjectDB       │
│  Tracking (ByteTrack)       EvidencePdf       ModelManager    │
│  Roi / Homography           ComplaintDraft                    │
│  ViolationRules                                               │
├─────────────────────────────────────────────────────────────┤
│    utils (GPU 偵測 · ffmpeg bin · logging · 設定)             │
└─────────────────────────────────────────────────────────────┘
```

**相依方向只能由上往下**。`ml/` 不 import `ui/`、不 import `core/`;`core/` 不 import `ui/`。這使 `ml/` 可在無 GUI 環境跑 batch、可單獨測試。

## 3. 主要資料流

1. 使用者選擇影片 → `ProjectWizard` 建立新的 `.zgproj` → 使用者在 `RoiEditor` 於第一幀畫斑馬線多邊形 → `Calibration` 輸入地面參考點的實際距離 → 儲存至專案檔。
2. 啟動分析 → `core.Pipeline` 在 `core.Worker` (QThread) 中跑:
   - **Stage 1 粗篩**:低 FPS(預設 1 fps)、輕量模型(`yolo11n`),掃出「行人 + 車輛同時出現在 ROI 附近」的可疑時段。
   - **Stage 2 精細**:對可疑時段,較高 FPS(預設 5–10 fps)與較大模型(`yolo11m`),加上 ByteTrack 建立軌跡,送入 `violation_rules` 判定。
3. 偵測結果即時寫入專案 SQLite,UI 透過 Qt Signal 顯示違規事件。
4. 使用者在 `ReviewDeck` 審查(採用/拒絕/調整時段)。
5. 使用者匯出 → `export.clip` 以 ffmpeg 無損剪輯 → `export.evidence_pdf` 產報告 → `export.complaint_draft` 產檢舉書草稿。

## 4. 執行緒模型

| Thread | 職責 |
|---|---|
| UI (main) | PySide6 event loop;只處理 UI 事件、Qt Signal 接收。 |
| Decoder | PyAV 解碼影片幀,寫入 bounded queue(避免把整段影片讀進記憶體)。 |
| Inference | ONNX Runtime 跑偵測 + ByteTrack;結果寫入第二個 queue。 |
| Writer | 把偵測/違規結果寫入 SQLite,並發 Qt Signal 通知 UI。 |

**取消機制**:每個 thread 都持有同一個 `threading.Event` cancel flag,每個迭代頂部檢查,檢查到就清 queue、關資源、離開。

**Checkpoint**:Writer thread 每 N 秒 flush 一次 SQLite 並更新 `project.progress_seconds`。重啟時 Pipeline 從 `progress_seconds` 繼續。

## 5. 專案檔 (.zgproj) 格式

實作上為一個**資料夾**(未來可考慮 zip 打包),內含:

```
my_crossing_2026-04-17.zgproj/
├── project.json          # 影片路徑、ROI、homography、設定、版本
├── project.db            # SQLite:偵測、追蹤、違規、使用者標註
├── thumbnails/           # 違規事件關鍵幀快取
└── exports/              # 使用者匯出的片段與報告
```

`project.json` 範例結構:

```json
{
  "version": 1,
  "video": { "path": "...", "sha256": "...", "duration_sec": 43200 },
  "roi": { "polygon": [[x, y], ...] },
  "homography": {
    "image_points": [[x, y], ...],
    "world_points_meters": [[x, y], ...],
    "matrix": [[...]]
  },
  "pipeline_settings": {
    "stage1_fps": 1,
    "stage2_fps": 5,
    "yield_distance_meters": 3.0
  },
  "progress": { "stage1_done": true, "stage2_seconds": 12345 }
}
```

`project.db` schema 初稿(待實作時調整):

- `detections(frame_idx, cls, bbox, conf)`
- `tracks(track_id, cls, start_frame, end_frame, path_json)`
- `violations(id, start_sec, end_sec, pedestrian_track_id, vehicle_track_id, evidence_frames, user_status, user_note)`
- `meta(key, value)`

## 6. ML 管線細節

### 6.1 偵測
- 模型:YOLOv8 / YOLO11(Ultralytics 預訓練 COCO 權重)
- 執行:轉換為 ONNX,透過 ONNX Runtime 推論
- Execution Provider 優先序:`CUDAExecutionProvider` → `DmlExecutionProvider`(DirectML,涵蓋 AMD/Intel/NVIDIA)→ `CPUExecutionProvider`

### 6.2 追蹤
- ByteTrack(或 Ultralytics 內建 `model.track()` 簡化整合)
- 輸出:每個 track_id 的 bbox 軌跡 + 類別

### 6.3 ROI 幾何
- 使用者畫的多邊形以 Shapely 做 point-in-polygon / intersection 判斷
- 「行人是否在斑馬線上」:行人 bbox 底邊中心是否在 ROI polygon 內
- 「車輛是否接近斑馬線」:車輛 bbox 底邊中心到 ROI polygon 最近點的**實際距離**(需 homography)

### 6.4 Homography 透視轉換
- 使用者在校正介面選 4 個以上地面點,並輸入該點相對於某原點的實際距離(公尺)
- OpenCV `cv2.findHomography` 計算 3×3 變換矩陣
- 後續所有「實際距離」查詢都經由此矩陣把影像座標換到世界座標

### 6.5 違規規則引擎 (`violation_rules.py`)
輸入:某一段時間內所有 track 的軌跡 + ROI + homography + 設定。

違規判定條件(草案,實作時可能調整,見 `legal-rules.md`):

```
若於某時間區間 [t0, t1] 內:
  存在 pedestrian_track P,使得 P 於該區間有幀位於 ROI 內,
  且同時存在 vehicle_track V,滿足:
    (a) V 與 P 之實際距離 ≤ yield_distance_meters(預設 3.0)
    (b) V 的速度未在接近 P 時降至 stop_threshold 以下
        (速度由連續幀的 bbox 位移 + homography + fps 推算)
則 (P, V, [t0, t1]) 為一筆違規候選事件。
```

## 7. 模型分發

- 安裝檔綁附 `yolo11n.onnx`(~12 MB)作為離線可用的後備。
- 首次啟動詢問使用者是否下載較大模型(`yolo11m.onnx`,~100 MB 級);存至 `%LOCALAPPDATA%\ZebraGuard\models\`。
- `storage.ModelManager` 負責:下載、SHA256 校驗、版本檢查、清理。
- CDN:TODO — 先用 GitHub Release,量大時評估 HuggingFace Hub 或 R2。

## 8. 設定與使用者資料位置

| 類型 | 位置 |
|---|---|
| 安裝目錄 | `%LOCALAPPDATA%\Programs\ZebraGuard\` (免管理員權限) |
| 使用者模型 | `%LOCALAPPDATA%\ZebraGuard\models\` |
| 使用者設定 | `%APPDATA%\ZebraGuard\settings.ini` (QSettings) |
| 日誌 | `%LOCALAPPDATA%\ZebraGuard\logs\` |
| 專案檔 | 使用者自選位置,預設「文件」資料夾 |

## 9. 尚未決定 (TODO)

- 依賴管理工具:`uv` vs `poetry` vs `pip-tools`
- 車牌 OCR 模型選型(Phase 2)
- 自動更新機制(Phase 2+)
- 國際化:預設 zh-TW,英文介面是否需要
