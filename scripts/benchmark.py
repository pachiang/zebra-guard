"""量測本機硬體上的偵測 fps,供 UI 顯示誠實 ETA。

Usage:
    python scripts/benchmark.py [--model PATH] [--frames N] [--ep cuda|dml|cpu]

目前為 placeholder。實作時要做:
    - 載入指定 .onnx 模型
    - 以同一張樣本圖片跑 N 次推論,計算 fps
    - 若指定 --ep,強制 ONNX Runtime execution provider
    - 輸出 JSON: {ep, fps, p50_ms, p95_ms, hardware}
    - 結果供設定頁顯示,並作為 ETA 計算依據
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resources/models/yolo11n.pt")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--ep", choices=["cuda", "dml", "cpu"])
    parser.parse_args(argv)

    print("[benchmark] 尚未實作。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
