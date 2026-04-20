# 模式 × Backend 架構

> 本文件是工程藍圖,比 `mvp.md` 低一層。重點在:app 有**兩種 mode**;其中 dashcam mode 底下有**可插拔的 crosswalk backend**。跨 session / 換 context 都該先讀這份,確認「哪條線活著、下一步做什麼」。

## 使用者看到的兩種 mode

| Mode | 影片情境 | 斑馬線來源 | 距離單位 | 硬體 |
|---|---|---|---|---|
| **Static** | 固定式攝影機 / 路口 CCTV | **使用者手繪 ROI**(+ 可選 homography) | 公尺(有 homography)或像素 | CPU 即可 |
| **Dashcam** | 行車記錄器 | **可選 backend**(見下) | 像素 | 依 backend 而定 |

Dashcam mode 底下的 **crosswalk backend** 插槽:

| Backend | 硬體需求 | 精度 | 狀態 |
|---|---|---|---|
| `mask2former` | **需 GPU**(swin-large) | 高 | ✅ 主線,UI 已串 |
| `yolo_seg` | CPU 即可 | 中(看權重品質) | 實驗中(本 branch) |
| (未來可加 SegFormer-b0、BiSeNet 等) | — | — | — |

## 設計原則

只換斑馬線來源,**下游 dilate / connected-components / 規則 / 聚合 / 事件 UI / 匯出全部共用一份程式碼**。這條規則是三線共存能長期維運的關鍵。

## 共用後段(三線必須維持相同行為)

- YOLO person/vehicle 偵測 + ByteTrack 追蹤(`ml/detection.py`)
- Ego vehicle 過濾(dashcam 專用,線 A 可關)
- Rider vs 行人過濾
- 同一幀規則:行人底部 strip 壓到 mask component X + 車輛底部 strip 壓到**同一** component X + 車輛有位移
- 事件聚合:`merge_gap_sec` 容忍度、`min_event_frames` 下限
- 專案檔(`.zgproj`)與 UI(review / export / complaint draft)

## CrosswalkSource 介面

```python
class CrosswalkSource(Protocol):
    """每幀回傳 labeled mask:背景=0,不同 crosswalk component 從 1 開始編號。

    回傳 shape (H, W) int32;H/W 必須等於輸入 frame 的影像尺寸。
    Source 自行決定快取:B 是每 N 幀重算、C 是每幀都跑、Static 永遠回同一張。
    """

    def get_labels(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray: ...

    def close(self) -> None: ...  # 釋放 GPU / 檔案資源
```

三個實作:

| 實作 | Mode | 快取策略 | 建構輸入 |
|---|---|---|---|
| `StaticRoiSource` | Static | 整段影片同一張 labeled mask(來自 polygon) | ROI polygon、(option) homography |
| `Mask2FormerSource` | Dashcam | 每 `mask_every` 幀重算,中間回傳上次結果 | model_name、device、`mask_imgsz` |
| `YoloSegSource` | Dashcam | 每幀都跑(YOLO-seg 夠快)或依 stride 跳 | `.pt` 權重路徑、`conf`、`imgsz` |

下游(dilate、connected-components、bottom-strip 規則、rider/ego 過濾、事件聚合)**完全共用一份程式碼**,放在 `ml/crosswalk/rules.py`(從 `scripts/zero_shot_detect.py` 搬過來)。

### 距離單位的例外(Static + homography)

靜態模式若使用者另外做了 homography 校正,規則層**仍用 mask**判「車輛底邊是否壓到斑馬線」,但額外把「車-人像素距離」經 homography 轉成公尺用於顯示與閾值比較。mask 判定維持三線一致,homography 只是加值資訊。

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
| `main` | Dashcam mode + Mask2Former backend + UI。任何動到 segmentation 的改動都要過 regression |
| `experiment/yolo-seg-crosswalk` | **目前的 branch**。抽 CrosswalkSource 介面 + 加 YoloSegSource。穩定後 squash merge 回 main |
| `experiment/static-mode`(未開) | Static mode 接線(UI 加 mode 選擇、ROI 編輯器、homography、用 StaticRoiSource) |

**保守原則**:
- 抽介面與加 backend 都在同一 branch 做;在 branch 內先跑 regression 確認主線輸出 byte-level 不變
- 支線先確認「YOLO-seg 能產生合理 events」為 merge 門檻;不要求與 Mask2Former 同等品質
- merge 前最後一次跑 regression,確認 mask2former backend 仍與 25/276/34.0s 基準一致

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

### Static mode(未接線,零件齊)
- `src/zebraguard/ml/crosswalk_detect.py` — 純 CV top-hat 條紋偵測(可半自動畫 ROI)
- `src/zebraguard/ml/{roi,homography,tracking,violation_rules}.py` — 舊管線
- `src/zebraguard/core/pipeline.py` — 舊編排
- `src/zebraguard/cli.py` — 舊 CLI
- `scripts/{roi_picker,detect_crosswalk,run_pipeline,smoke_detect}.py` — 舊工具
- `tests/ml/*` — 還在通過的單元測試(別砍)

### 插拔式 backend(本 branch 正在建)
- `src/zebraguard/ml/crosswalk/` 子套件
  - `base.py` — `CrosswalkSource` Protocol
  - `mask2former.py` — 從 `scripts/zero_shot_detect.py` 抽出
  - `yolo_seg.py` — 新建;需使用者提供 YOLO-seg 斑馬線 `.pt` 權重
  - `static_roi.py` — 未來再寫(Static mode 用)
- `src/zebraguard/ml/crosswalk/rules.py`(規劃中)— 下游共用規則

## 下一步工作清單(按順序)

- [x] **0. Commit 目前 main 狀態**(UI + docs 更新)— commit `aff70c0`
- [x] **1. 開 branch** `experiment/yolo-seg-crosswalk`
- [ ] **2. 抽介面**:建立 `ml/crosswalk/__init__.py`、`ml/crosswalk/base.py`(Protocol)、`ml/crosswalk/mask2former.py`(把 `segment_zebra` + dilate + components 搬過來、包成 CrosswalkSource)
- [ ] **3. 主線切換**:`scripts/zero_shot_detect.py` 改成用 `Mask2FormerSource`;規則/聚合留在原檔(第一步只動 segmentation 那段)
- [ ] **4. Regression**:用 taiwan_yield_720p.mp4 跑完整 600s,比對 events 數與時間戳跟前面 25/276/34.0s 對得上
- [ ] **5. 加 YoloSegSource**:`ml/crosswalk/yolo_seg.py`;需要使用者提供或下載 YOLO-seg 斑馬線 `.pt`
- [ ] **6. Pipeline backend flag**:`scripts/zero_shot_detect.py` 加 `--crosswalk-backend {mask2former,yolo_seg}`
- [ ] **7. Project + UI 串起 mode/backend**:新專案 wizard 問 mode(dashcam/static),dashcam 時再問 backend;Project 記錄以便重跑
- [ ] **8. 文件回填**:結果寫回本檔的「目前主線行為」與 `architecture.md`

### 給「下一個 session」的 session-handoff checklist

接手前請確認:
1. `git branch --show-current` — 目前 branch 是 `experiment/yolo-seg-crosswalk`;主線上**不該有動到 segmentation 的未 commit 變更**
2. 讀 `docs/pipelines.md`(本檔) + `docs/mvp.md`
3. 讀本檔「下一步工作清單」,找出第一個未打勾項目開始做
4. 動任何會影響 mask 輸出的改動前,先跑「主線行為」區塊的 regression 指令存一份 baseline;動完再跑一次比對
5. 若接手 Static mode:先 grep `scripts/roi_picker.py` 與 `ml/crosswalk_detect.py`;UI 需要加 mode 選擇

### 已做決定(避免重複討論)

- Static mode 不在 MVP 範圍;是三線架構的**預留插槽**,UI 會有 mode 選擇,但實作晚於 YOLO-seg
- Mainline Mask2Former 效果可接受,不改參數也不改模型。YOLO-seg 是為了讓無 GPU 使用者可用;品質可略差
- UI 走同一套 review / export;Project 新增 `mode` + `crosswalk_backend` 兩欄以便重現
- `core/worker.py` 目前以 importlib 載入 `scripts/zero_shot_detect.py`。抽介面後,worker 改成直接 import `zebraguard.ml.crosswalk` + 本地的 pipeline runner;`scripts/zero_shot_detect.py` 保留為 CLI shim
- YOLO-seg 的 `.pt` 權重**不綁附在 repo 內**(避免版權與檔案大小),使用者 / 開發者自行放到 `resources/models/` 或在 preset 指定路徑
