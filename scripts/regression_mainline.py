"""主線 Mask2Former backend byte-level regression。

用法:
  # 建 baseline(只在 main 上、抽介面之前跑一次)
  python scripts/regression_mainline.py --video taiwan_yield_720p.mp4 \
      --out-baseline regression_baseline.json

  # 跑目前 code 並比對
  python scripts/regression_mainline.py --video taiwan_yield_720p.mp4 \
      --baseline regression_baseline.json

比對忽略 params 區塊(因為新增了 backend 欄位),只比 events + frame_hits。
預期:23-27 events 附近、270-280 frame_hits(與 v7 baseline 一致)。

等價性通過的條件:events 陣列 byte-for-byte 相同。不相同則視為 regression。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _v7_command(video: Path, out_json: Path) -> list[str]:
    """與 presets/v7_baseline_cmd.sh 完全同步的參數。"""
    return [
        sys.executable,
        str(ROOT / "scripts" / "zero_shot_detect.py"),
        "--video", str(video),
        "--out", str(out_json),
        "--stride", "3",
        "--mask-every", "5",
        "--dilate-px", "20",
        "--prox-px", "0",
        "--moving-px", "2.0",
        "--conf", "0.20",
        "--imgsz", "640",
        "--merge-gap-sec", "0.6",
        "--min-event-frames", "2",
        "--ego-bottom-px", "4",
        "--ego-min-width-frac", "0.40",
        "--ego-min-height-frac", "0.25",
        "--max-seconds", "600",
        "--min-mask-area-frac", "0.005",
        "--rider-contain-thresh", "0.40",
        "--rider-foot-margin-px", "12",
        "--person-conf", "0.35",
        "--mask-imgsz", "640",
        "--crosswalk-backend", "mask2former",
    ]


def _strip_params(report: dict) -> dict:
    """只保留 regression 要比對的欄位(去掉 params 區塊的 backend/weights 差異)。"""
    return {
        "events": report.get("events", []),
        "frame_hits": report.get("frame_hits"),
        "frame_count": report.get("frame_count"),
    }


def _run(video: Path, out_json: Path) -> dict:
    cmd = _v7_command(video, out_json)
    print("[regression] 執行:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[regression] 管線 exit={result.returncode};放棄比對", file=sys.stderr)
        sys.exit(result.returncode)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-baseline", type=Path, default=None,
                    help="若設定:跑一次 pipeline,把結果寫入這裡當作 baseline,然後離開")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline JSON 的位置;用於比對當前 code 的輸出")
    ap.add_argument("--current-out", type=Path,
                    default=ROOT / "regression_current.json",
                    help="本次 run 的輸出位置(預設於 repo 根)")
    args = ap.parse_args()

    if (args.out_baseline is None) == (args.baseline is None):
        print("必須擇一:--out-baseline(建 baseline) 或 --baseline(比對)",
              file=sys.stderr)
        return 2

    if args.out_baseline is not None:
        _run(args.video, args.out_baseline)
        print(f"[regression] baseline 寫入:{args.out_baseline}")
        return 0

    cur = _run(args.video, args.current_out)
    base = json.loads(args.baseline.read_text(encoding="utf-8"))

    a = _strip_params(base)
    b = _strip_params(cur)
    if a == b:
        n = len(a["events"])
        print(f"[regression] ✅ PASS — events / frame_hits / frame_count byte-equal ({n} events)")
        return 0

    # 粗粒度 diff 訊息
    print("[regression] ❌ FAIL — 輸出不一致")
    print(f"  baseline events = {len(a['events'])}   frame_hits = {a['frame_hits']}")
    print(f"  current  events = {len(b['events'])}   frame_hits = {b['frame_hits']}")
    # 列前兩筆 event 的 start_frame 比對
    for i in range(min(len(a["events"]), len(b["events"]))):
        ea, eb = a["events"][i], b["events"][i]
        if ea != eb:
            print(f"  event[{i}] 不同:")
            print(f"    baseline: {ea}")
            print(f"    current : {eb}")
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
