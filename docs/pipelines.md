# 三條斑馬線來源線路

> 本文件是工程藍圖,比 `mvp.md` 低一層;重點在:三條斑馬線來源要共存於同一 codebase,如何不互相干擾。跨 session / 換 context 都該先讀這份,確認「哪條線活著、下一步做什麼」。

## 背景 / 為何三線

ZebraGuard 判定「車輛未禮讓」,最關鍵的資料就是「斑馬線在畫面中的位置」。這件事目前有三種取得方式,各有最佳情境,短期都有價值:

| 代號 | 斑馬線來源 | 影片情境 | 硬體需求 | 距離單位 | 狀態 |
|---|---|---|---|---|---|
| **A** | 使用者手繪 ROI(+ 可選 homography) | 固定式攝影機 / 路口 CCTV | CPU 即可 | 公尺(經 homography) | 零件齊全,未接線 |
| **B** | **Mask2Former 每幀自動分割** | 行車記錄器(dashcam) | **需 GPU**(swin-large) | 像素近似 | ✅ **主線**,UI 已串接 |
| **C** | **YOLO-seg 斑馬線模型**(fine-tuned) | 行車記錄器(dashcam) | CPU 即可 | 像素近似 | 實驗中(branch) |

三條線的設計是「只換 crosswalk 來源,後段全部共用」。不是三個獨立 pipeline。

## 共用後段(三線必須維持相同行為)

- YOLO person/vehicle 偵測 + ByteTrack 追蹤(`ml/detection.py`)
- Ego vehicle 過濾(dashcam 專用,線 A 可關)
- Rider vs 行人過濾
- 同一幀規則:行人底部 strip 壓到 mask component X + 車輛底部 strip 壓到**同一** component X + 車輛有位移
- 事件聚合:`merge_gap_sec` 容忍度、`min_event_frames` 下限
- 專案檔(`.zgproj`)與 UI(review / export / complaint draft)

## CrosswalkSource 介面(設計中)

```python
class CrosswalkSource(Protocol):
    """每幀回傳 labeled mask:背景=0,不同 crosswalk component 從 1 開始編號。

    回傳 shape (H, W) int32;H/W 必須等於輸入 frame 的影像尺寸。
    同一 source 在連續幀間可以快取自己的成本(例如 B 線每 N 幀重算、其餘幀重用)。
    """

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray: ...
```

三個實作:

| 實作 | 快取策略 | 特殊資料 |
|---|---|---|
| `StaticRoiSource`(線 A) | 整段影片同一張 mask | 建構時傳 polygon;可選擇傳 homography |
| `Mask2FormerSource`(線 B) | 每 `mask_every` 幀重算,中間回傳上次結果 | GPU device |
| `YoloSegSource`(線 C) | 每幀都跑(YOLO-seg 夠快) | `.pt` 權重路徑 |

下游(dilate、connected-components、規則、聚合)**完全共用一份程式碼**。

### 距離單位的例外

線 A 若開啟 homography,規則層可以把車輛距離換成公尺,但**不是把 mask 棄掉**——仍然用 mask 判「車輛底邊是否壓到斑馬線」,只是另外把「車-人像素距離」經 homography 轉成公尺用於顯示與閾值比較。這讓三條線在規則觸發上保持一致,線 A 的 homography 只是加值資訊。

## 目前主線行為(不可破壞)

主線 = 線 B,v7 preset。`scripts/presets/v7_baseline_params.json` + `scripts/zero_shot_detect.py` + UI 串接 via `core/worker.py`。

**Byte-level regression 基準**:

```
影片:taiwan_yield_720p.mp4 首 600 秒
v7 結果:25 events / 276 hits / 34.0s flagged (5.7% of 10min)
```

任何重構或新增線路的 PR,都要能在原 preset 下跑出**相同**的 events(id、timestamps、ped/veh_track_ids)。Regression 測試建議跑:

```bash
.venv/Scripts/python.exe scripts/zero_shot_detect.py \
  --video taiwan_yield_720p.mp4 --out /tmp/new.json \
  --stride 3 --mask-every 5 --dilate-px 20 --prox-px 0 \
  --moving-px 2.0 --conf 0.20 --imgsz 640 --merge-gap-sec 0.6 \
  --min-event-frames 2 --max-seconds 600 --min-mask-area-frac 0.005 \
  --rider-contain-thresh 0.40 --rider-foot-margin-px 12 \
  --person-conf 0.35 --mask-imgsz 640
# 然後 diff 新舊 events(忽略 params 區塊)
```

## 分支策略

| 分支 | 用途 |
|---|---|
| `main` | 主線 (B) + UI + 共用後段。任何動到主線的改動都要過 regression |
| `experiment/yolo-seg-crosswalk` | 線 C 的 CrosswalkSource 抽象 + YOLO-seg 實作。**穩定後 squash merge 回 main** |
| `experiment/static-camera`(未開) | 線 A 再次接線(homography + 手繪 ROI + 舊 violation_rules.py 的再活化) |

**保守原則**:
- 支線先在自己 branch 活,驗證「新 backend 能產出 event」為止
- 要 merge 回 main 之前,先做抽象重構(把 `segment_zebra` 抽成 `Mask2FormerSource`),regression 通過後再合;避免主線被打擾

## 檔案索引(目前狀態)

### 共用
- `src/zebraguard/ml/detection.py` — YOLO + ByteTrack 包裝
- `src/zebraguard/ml/types.py` — 共用 dataclass(Detection / Track / PipelineConfig)
- `src/zebraguard/core/project.py` — `.zgproj` 模型;`events` 表(v7+ 新格式)與 `violations` 表(舊 homography 格式)兩套並存
- `src/zebraguard/core/worker.py` — QThread 包 pipeline
- `src/zebraguard/ui/` — 整套 UI
- `src/zebraguard/export/` — 片段 / 關鍵幀 / 檢舉書

### 線 B(主線)目前實作位置
- `scripts/zero_shot_detect.py` — pipeline 本體(含 `segment_zebra` / rule / 聚合)
- `scripts/presets/v7_baseline_*` — 參數
- `core/worker.py` 透過 importlib 載入 `zero_shot_detect.py`

### 線 A(未接線,零件齊)
- `src/zebraguard/ml/crosswalk_detect.py` — 純 CV top-hat 條紋偵測(可半自動畫 ROI)
- `src/zebraguard/ml/{roi,homography,tracking,violation_rules}.py` — 舊管線
- `src/zebraguard/core/pipeline.py` — 舊編排
- `src/zebraguard/cli.py` — 舊 CLI
- `scripts/{roi_picker,detect_crosswalk,run_pipeline,smoke_detect}.py` — 舊工具
- `tests/ml/*` — 還在通過的單元測試(別砍)

### 線 C(實驗中)
- **尚未建立**。將置於 `src/zebraguard/ml/crosswalk/` 子套件。

## 下一步工作清單(按順序)

- [ ] **0. Commit 目前 main 狀態**(UI + docs 更新)
- [ ] **1. 開 branch** `experiment/yolo-seg-crosswalk`
- [ ] **2. 抽介面**:建立 `ml/crosswalk/__init__.py`、`ml/crosswalk/base.py`(Protocol)、`ml/crosswalk/mask2former.py`(把 `segment_zebra` + dilate + components 搬過來、包成 CrosswalkSource)
- [ ] **3. 主線切換**:`scripts/zero_shot_detect.py` 改成用 `Mask2FormerSource`;規則/聚合留在原檔(第一步只動 segmentation 那段)
- [ ] **4. Regression**:用 taiwan_yield_720p.mp4 跑完整 600s,比對 events 數與時間戳跟前面 25/276/34.0s 對得上
- [ ] **5. 加 line C**:`ml/crosswalk/yolo_seg.py`;先以 Roboflow 上找到的 pretrained weights 測試,評估品質
- [ ] **6. Pipeline backend flag**:`scripts/zero_shot_detect.py` 加 `--crosswalk-backend {mask2former,yolo_seg}`
- [ ] **7. UI 暫不動**:等 YOLO-seg 品質確認再決定要不要開 backend 選擇 UI
- [ ] **8. 文件回填**:結果寫回本檔的「目前主線行為」與 `architecture.md`

### 給「下一個 session」的 session-handoff checklist

接手前請確認:
1. `git branch --show-current` 看目前在哪條 branch;主線上**不該有動到 segmentation 的未 commit 變更**
2. 讀 `docs/pipelines.md`(本檔) + `docs/mvp.md`
3. 讀本檔「下一步工作清單」,找出第一個未打勾項目開始做
4. 若接手線 C:優先跑 regression(步驟 4)確認主線未偏
5. 若接手線 A:先 grep `scripts/roi_picker.py` 與 `ml/crosswalk_detect.py`;舊管線要先讓 UI 能選擇「我要手繪 ROI」模式

### 已做決定(避免重複討論)

- 不做靜態攝影機的 MVP,只做 dashcam(見 `mvp.md`)。線 A 之後才接;三線架構是**未來擴充**的預留。
- 主線 v7 效果可接受,不改參數也不改模型。線 C 只是為了在無 GPU 機器上能跑;品質可能略差可以接受。
- UI 目前綁定線 B;線 C 進來後仍走同一 UI,只是 Project 內多記一個 `crosswalk_backend` 欄位以便重現。
- `core/worker.py` 用 importlib 載入 `scripts/zero_shot_detect.py`——這是暫時的,抽介面後 zero_shot_detect 會被搬進 `ml/zero_shot_pipeline.py`,worker 直接 import。本檔步驟 2-3 會觸及此搬家。
