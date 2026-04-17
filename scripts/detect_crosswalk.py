"""自動偵測斑馬線 → 產生 roi.json + homography.json + 預覽圖。

Usage:
    python scripts/detect_crosswalk.py --video V.mp4 [--frame 0] \\
        [--out-dir .] [--stripe-width-m 0.4] [--stripe-period-m 1.0]

也可以直接吃圖檔:
    python scripts/detect_crosswalk.py --image frame.png ...

輸出:
    roi.json
    homography.json
    crosswalk_preview.png  — 偵測結果疊圖
"""

from __future__ import annotations

import argparse
import json
import sys
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

from zebraguard.ml.crosswalk_detect import (  # noqa: E402
    DEFAULT_CROSSING_LENGTH_M,
    TW_STRIPE_PERIOD_M,
    TW_STRIPE_WIDTH_M,
    binarise,
    detect,
    draw_preview,
    find_stripe_candidates,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"無法讀取 {args.image}")
        return frame
    if args.video is None:
        raise SystemExit("請指定 --video 或 --image")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟 {args.video}")
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"讀取 frame {args.frame} 失敗")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--image", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--stripe-width-m", type=float, default=TW_STRIPE_WIDTH_M,
                        help=f"條紋寬度 m(預設台灣 {TW_STRIPE_WIDTH_M})")
    parser.add_argument("--stripe-period-m", type=float, default=TW_STRIPE_PERIOD_M,
                        help=f"條紋週期 條紋+間隔 m(預設 {TW_STRIPE_PERIOD_M})")
    parser.add_argument("--crossing-length-m", type=float, default=DEFAULT_CROSSING_LENGTH_M,
                        help=f"行走方向長度 m,沿條紋長軸(預設 {DEFAULT_CROSSING_LENGTH_M})")
    parser.add_argument("--debug", action="store_true",
                        help="輸出 binary / candidates 中間圖")
    args = parser.parse_args(argv)

    frame = _load_frame(args)
    h, w = frame.shape[:2]
    _log(f"Frame: {w}x{h}")

    if args.debug:
        binary = binarise(frame)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_dir / "debug_binary.png"), binary)
        candidates = find_stripe_candidates(binary)
        dbg = frame.copy()
        for s in candidates:
            cv2.drawContours(dbg, [s.corners.astype(np.int32)], 0, (0, 200, 255), 1)
        cv2.imwrite(str(args.out_dir / "debug_candidates.png"), dbg)
        _log(f"候選條紋:{len(candidates)}  (debug 圖存在 {args.out_dir})")

    result = detect(
        frame,
        stripe_width_m=args.stripe_width_m,
        stripe_period_m=args.stripe_period_m,
        crossing_length_m=args.crossing_length_m,
    )

    candidates = find_stripe_candidates(binarise(frame))
    preview = draw_preview(frame, candidates, result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.out_dir / "crosswalk_preview.png"
    cv2.imwrite(str(preview_path), preview)
    _log(f"預覽:{preview_path}")

    if result is None:
        _log("找不到符合的斑馬線群組。")
        return 1

    _log(f"命中!")
    _log(f"  條紋數量:        {len(result.stripes)}")
    _log(f"  平均條紋厚度:    {result.avg_thickness_px:.1f} px  (= {args.stripe_width_m} m)")
    _log(f"  跨越方向長度:    {result.crossing_width_m:.2f} m  "
         f"((N-1)*period + stripe_width)")
    _log(f"  行走方向長度:    {result.crossing_length_m:.2f} m  (從參數)")
    _log(f"  Homography 對應: {len(result.image_points)} 組(每條條紋 2 端點)")

    roi_path = args.out_dir / "roi.json"
    hom_path = args.out_dir / "homography.json"
    roi_path.write_text(json.dumps(
        {"polygon": [list(p) for p in result.roi_image_corners.tolist()]},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    hom_path.write_text(json.dumps(
        {
            "image_points": result.image_points,
            "world_points_meters": result.world_points_meters,
        },
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    _log(f"寫入 {roi_path}")
    _log(f"寫入 {hom_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
