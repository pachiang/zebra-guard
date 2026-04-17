"""A 方案:剪 roadsafety.mp4 到穩定片段 + 估計 ROI/homography + 跑完整 pipeline。

動機:`roadsafety.mp4` 是多場景剪輯,全片沒辦法用單一 ROI。但 frame 150–210
之間約 60 幀(2 秒)是日景、藍車經過斑馬線 + 紅貨車路口 的穩定片段,
我們裁下半畫面並估計 ROI/homography 做端到端 demo。

  ROI/homography 是我視覺估計的,精度當 PoC 看就好。要正式判違規
  應該用 `scripts/roi_picker.py` 在這段影片上實際標註。

本腳本把所有步驟做在同一個 Python process 內,避免每次呼叫 CLI 都觸發
torch 冷啟(Defender 掃 DLL)。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SOURCE_VIDEO = ROOT / "roadsafety.mp4"
FIXTURES = ROOT / "tests" / "fixtures"
CLIP_VIDEO = FIXTURES / "roadsafety_clip.mp4"
CLIP_FRAME_PNG = FIXTURES / "roadsafety_clip_frame0.png"
ROI_JSON = FIXTURES / "roadsafety_roi.json"
HOM_JSON = FIXTURES / "roadsafety_hom.json"
PROJECT_PATH = ROOT / "demo.zgproj"

# 要擷取的幀區間(原片 30 fps)
FRAME_START = 150
FRAME_END = 210

# 下半畫面:y 座標範圍
# roadsafety.mp4 是 1080x1920(portrait),下半約在 y=960 到 y=1920
CROP_Y0 = 960
CROP_Y1 = 1920  # 含片尾標籤條,保留


def _log(msg: str) -> None:
    print(msg, flush=True)


def step_1_clip_and_crop() -> tuple[int, int, float, np.ndarray]:
    """把 SOURCE_VIDEO 的 frame 150-210 裁下半部分寫成 CLIP_VIDEO。"""
    _log(f"[1/5] 剪片 + 裁切 → {CLIP_VIDEO.relative_to(ROOT)}")
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟 {SOURCE_VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("讀取起始幀失敗")
    h, w = first.shape[:2]
    crop_h = CROP_Y1 - CROP_Y0
    _log(f"      原片 {w}x{h} @ {fps:.2f}fps;裁成 {w}x{crop_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(CLIP_VIDEO), fourcc, fps, (w, crop_h))

    cropped_first = first[CROP_Y0:CROP_Y1, :, :]
    writer.write(cropped_first)
    cv2.imwrite(str(CLIP_FRAME_PNG), cropped_first)

    idx = FRAME_START + 1
    while idx <= FRAME_END:
        ok, f = cap.read()
        if not ok:
            break
        writer.write(f[CROP_Y0:CROP_Y1, :, :])
        idx += 1
    cap.release()
    writer.release()
    _log(f"      共 {idx - FRAME_START} 幀寫入")
    _log(f"      首幀存成 {CLIP_FRAME_PNG.relative_to(ROOT)}")
    return w, crop_h, fps, cropped_first


def step_2_write_roi_and_homography(w: int, h: int) -> None:
    """根據視覺估計寫 roi.json 與 homography.json。

    相對座標:以裁切後的畫面(下半部分)為準,不是原片全幀。
    下半畫面中的斑馬線大致位置(手測):
      - 斑馬線左上(遠處、靠左側):pixel (80, 140), 世界 (0, 0)  m
      - 斑馬線右上(遠處、靠右):  pixel (720, 90), 世界 (6, 0)  m
      - 斑馬線右下(近處、靠右):  pixel (880, 560), 世界 (6, 3) m
      - 斑馬線左下(近處、靠左):  pixel (30, 650), 世界 (0, 3) m
    假設斑馬線 6m 寬 × 3m 長(英式典型 zebra crossing)。
    """
    _log(f"[2/5] 寫 ROI + Homography(視覺估計)")
    _ = (w, h)  # 目前未用;未來可做自動驗證
    import json

    # 斑馬線在畫面左下,對角方向。估計的 4 個角:
    #   P1 (80, 430)  — 斑馬線遠端左上(在 traffic island 前)
    #   P2 (520, 400) — 斑馬線遠端右上(接近 traffic island)
    #   P3 (600, 720) — 斑馬線近端右下(中央島右緣往下)
    #   P4 (0,  820)  — 斑馬線近端左下(畫面左緣被裁切)
    # 假設斑馬線 5m 寬(橫向)x 3m 深(行走方向)
    roi = {
        "polygon": [[80, 430], [520, 400], [600, 720], [0, 820]],
    }
    hom = {
        "image_points": [[80, 430], [520, 400], [600, 720], [0, 820]],
        "world_points_meters": [[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0]],
    }
    ROI_JSON.write_text(json.dumps(roi, ensure_ascii=False, indent=2), encoding="utf-8")
    HOM_JSON.write_text(json.dumps(hom, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"      {ROI_JSON.relative_to(ROOT)}")
    _log(f"      {HOM_JSON.relative_to(ROOT)}")


def step_3_annotate_frame(frame: np.ndarray) -> None:
    """把 ROI + homography 頂點畫在裁切後的首幀上,供人目視驗證。"""
    import json

    _log(f"[3/5] 把 ROI 疊到首幀上目視確認")
    roi = json.loads(ROI_JSON.read_text(encoding="utf-8"))
    pts = np.array(roi["polygon"], dtype=np.int32)
    img = frame.copy()
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.putText(img, f"P{i + 1}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"P{i + 1}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    out = FIXTURES / "roadsafety_clip_roi_overlay.png"
    cv2.imwrite(str(out), img)
    _log(f"      {out.relative_to(ROOT)}")


def step_4_run_pipeline() -> None:
    """跑完整 ZebraGuard pipeline 並寫入 .zgproj。"""
    _log(f"[4/5] 跑 pipeline(torch 首次 import 會慢,耐心等)")
    t0 = time.monotonic()

    from zebraguard.core.pipeline import run_pipeline
    from zebraguard.core.project import Project
    from zebraguard.ml.homography import Homography
    from zebraguard.ml.roi import RoiPolygon
    from zebraguard.ml.types import PipelineConfig

    _log(f"      ({time.monotonic() - t0:.1f}s) 匯入完成")

    roi = RoiPolygon.load(ROI_JSON)
    hom = Homography.load(HOM_JSON)
    cfg = PipelineConfig(
        yield_distance_m=3.0,
        stop_threshold_kmh=5.0,
        min_pedestrian_frames=3,
        model_name=str(ROOT / "resources" / "models" / "yolo11n.pt"),
    )

    def on_progress(cur: int, total: int) -> None:
        pct = 100.0 * cur / total if total else 0.0
        _log(f"      進度 {cur}/{total} ({pct:5.1f}%)")

    _log(f"      開始追蹤 {CLIP_VIDEO.relative_to(ROOT)}...")
    t1 = time.monotonic()
    result = run_pipeline(
        video_path=CLIP_VIDEO,
        roi=roi,
        homography=hom,
        config=cfg,
        on_progress=on_progress,
    )
    _log(f"      追蹤完成 ({time.monotonic() - t1:.1f}s)")
    _log(f"      影片: {result.video.fps:.2f} fps, {result.video.frame_count} frames")
    _log(f"      Tracks: {result.n_tracks}")
    _log(f"      違規候選: {len(result.violations)}")
    for i, v in enumerate(result.violations):
        _log(f"        #{i + 1}: ped={v.pedestrian_track_id} veh={v.vehicle_track_id}  "
             f"t=[{v.start_sec:.2f}–{v.end_sec:.2f}]s  "
             f"dist={v.min_distance_m:.2f}m  speed={v.min_vehicle_speed_kmh:.1f}km/h")

    _log(f"[5/5] 寫入 .zgproj 專案 → {PROJECT_PATH.relative_to(ROOT)}")
    if PROJECT_PATH.exists():
        _log(f"      既有專案已存在,覆蓋...")
        import shutil

        shutil.rmtree(PROJECT_PATH)
    project = Project.create(PROJECT_PATH, CLIP_VIDEO)
    with project:
        project.save_roi(roi.to_json())
        project.save_homography(hom.to_json())
        from dataclasses import asdict

        cfg_dict = asdict(cfg)
        cfg_dict["vehicle_classes"] = sorted(cfg.vehicle_classes)
        project.save_config(cfg_dict)
        project.save_violations(result.violations)
        project.update_progress("analyzed", violations=len(result.violations))
    _log(f"      project.json + project.db 已寫")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-only", action="store_true",
        help="只跑 step 1-3(剪片、ROI、疊圖),不跑 pipeline / 不等 torch",
    )
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not SOURCE_VIDEO.exists():
        _log(f"找不到 {SOURCE_VIDEO}")
        return 2

    w, h, fps, first_frame = step_1_clip_and_crop()
    step_2_write_roi_and_homography(w, h)
    step_3_annotate_frame(first_frame)
    if args.preview_only:
        _log("")
        _log("預覽模式完成。確認 roadsafety_clip_roi_overlay.png,然後"
             "重跑不帶 --preview-only。")
        return 0
    step_4_run_pipeline()
    _log("")
    _log("完成。檢查 tests/fixtures/roadsafety_clip_roi_overlay.png 看 ROI 疊圖,"
         "以及 demo.zgproj/ 看結果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
