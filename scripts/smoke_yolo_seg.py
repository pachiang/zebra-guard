"""YoloSegSource 煙霧測試:抽 N 幀、跑 mask、儲存可視化。

Usage:
    python scripts/smoke_yolo_seg.py \
        --video taiwan_yield_720p.mp4 \
        --weights resources/models/zebra_yolov8n_seg.pt \
        --frames 5,120,300,600,1500 \
        --out-dir yolo_seg_smoke/

產出每一指定幀的 3 張圖:原圖 + 標註版(binary mask 半透明綠)+ 標註版
(component 分顏色)。可用於人工檢查 YOLO-seg 權重對該影片的品質。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zebraguard.ml.crosswalk.yolo_seg import YoloSegConfig, YoloSegSource  # noqa: E402


PALETTE = [
    (0, 255, 0), (0, 200, 255), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 255, 0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--frames", default="30,120,300,600,1200",
                    help="逗號分隔的 frame index")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "yolo_seg_smoke")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dilate-px", type=int, default=20)
    args = ap.parse_args()

    frame_ids = [int(s) for s in args.frames.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = YoloSegSource(
        YoloSegConfig(
            weights=str(args.weights),
            conf=args.conf,
            imgsz=args.imgsz,
            dilate_px=args.dilate_px,
            infer_every=1,
            min_mask_area_frac=0.0,  # smoke 不過濾,要看原始輸出
        )
    )
    print(f"[load] {args.weights}")
    t0 = time.monotonic()
    src._ensure_loaded()
    print(f"[load] done in {time.monotonic() - t0:.2f}s")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[video] {args.video.name}: {total} frames @ {fps:.1f} fps")

    for fid in frame_ids:
        if fid >= total:
            print(f"[skip] frame {fid} > total {total}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            print(f"[skip] cannot read frame {fid}")
            continue

        t_inf = time.monotonic()
        labels = src.get_labels(frame, fid)
        dt = time.monotonic() - t_inf

        n_comp = int(labels.max())
        area = int((labels > 0).sum())
        area_frac = 100.0 * area / labels.size
        print(f"[frame {fid:>5d}]  {dt*1000:6.1f} ms   "
              f"components={n_comp}   area={area_frac:5.2f}%")

        # 原圖
        cv2.imwrite(str(args.out_dir / f"f{fid:05d}_raw.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Binary overlay
        overlay = frame.copy()
        mask = (labels > 0).astype(np.uint8)
        overlay[mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        cv2.putText(blended, f"f={fid} t={fid/fps:.1f}s  comps={n_comp}  "
                    f"area={area_frac:.2f}%  {dt*1000:.0f}ms",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"f{fid:05d}_mask.jpg"), blended,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Per-component colored
        comp_vis = frame.copy()
        ov = comp_vis.copy()
        for lab in range(1, n_comp + 1):
            color = PALETTE[(lab - 1) % len(PALETTE)]
            ov[labels == lab] = color
        comp_vis = cv2.addWeighted(ov, 0.5, comp_vis, 0.5, 0)
        cv2.imwrite(str(args.out_dir / f"f{fid:05d}_comps.jpg"), comp_vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()
    src.close()
    print(f"[done] 輸出於 {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
