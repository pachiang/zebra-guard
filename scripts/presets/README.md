# Presets — known-good configs

Parameter snapshots frozen at specific points so later tweaks can be compared and reverted.

| Preset | Source run | Result | Notes |
|---|---|---|---|
| `v5_baseline` | `zs_taiwan10_v5.json` (2026-04-19) | 31 events / 322 hits / 6.8% of 10min flagged | User-validated as good after v2→v5 tightening: strict mask rule, 0.5s mask refresh, 3-criterion rider filter, person conf 0.35. |
| `v7_baseline` | `zs_taiwan10_v7.json` (2026-04-19) | 25 events / 276 hits / 5.7% of 10min flagged | Adds same-component rule (fixes cross-crosswalk false alarms like the 09:24 taxi), bottom-strip overlap (handles vehicle self-occlusion), `mask_imgsz=640` (lower GPU heat), and removes the stale-mask fallback that caused ghost zebras at the bottom of the frame. GPU peak ~86°C (v6 was 88°C). Used by `crosswalk_backend=mask2former`. |
| **`yolo_seg_initial`** ✨ | 2026-04-20 首次 yolo_seg 跑測 | 29 events / 255 hits / 31.1s flagged | **yolo_seg backend 的預設** — 沿用 v7 數字 + YOLO 預設信心 0.25。**最原版、未特別針對 yolo_seg 調整**,作為 baseline of baselines。 |
| `yolo_seg_baseline` | 2026-04-21 tuned 版 | TBD(比 initial 更多小斑馬線 hits) | 針對「抓遠處小斑馬線」調整:`mask_every=2`、`min_mask_area_frac=0.001`、`yolo_seg_conf=0.20`、`yolo_seg_min_component_px=80`。 |
| `yolo_seg_tuned` | `yolo_seg_10min_me3_mma0.001_ysc0.15_imgsz640_comp80.json` (2026-04-21) | 27 events / 279 hits / 36.1s flagged | 更激進版 — `mask_every=3`、`yolo_seg_conf=0.15`。實測幾乎等於 Mask2Former v7 baseline(276/25/34.0)。適合想達到 M2F 等級但無 GPU 的使用者。 |

## Files per preset

- `<name>_params.json` — exact values for every `zero_shot_detect.py` CLI flag.
- `<name>_cmd.sh` — runnable bash that invokes the script with those flags. `bash scripts/presets/<name>_cmd.sh <video> <out.json> <annotated.mp4>`.

## Restoring a preset

Either run the `.sh` directly, or pass the flags from `_params.json` manually. The current `zero_shot_detect.py` reads every flag listed, so no script edit is required to reproduce an old result.

If a future code change breaks a preset's behavior (e.g., default for a flag changes), re-run the preset and compare the `params` block inside the resulting `*.json` against `<name>_params.json` — any diff flags the incompatibility.
