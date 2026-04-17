"""下載 YOLO11 模型到 resources/models/。

用 urllib 從 Ultralytics 的 GitHub assets release 直接抓,不依賴
ultralytics 套件本身(避免為了下載而安裝 torch ~2 GB)。

Usage:
    python scripts/download_models.py                     # 預設下載 yolo11n
    python scripts/download_models.py --model yolo11m     # 指定大小
    python scripts/download_models.py --all               # 全部大小
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Ultralytics 把 YOLO 權重放在 assets repo 的 release tag 下。
# v8.3.0 是 YOLO11 發佈時綁定的 tag,檔案長期存在。
_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

SIZES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "zebraguard-downloader"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total > 0:
                    pct = int(100 * got / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"\r  {pct:3d}%  {_human_size(got)}/{_human_size(total)}",
                              end="", flush=True)
                        last_pct = pct
    tmp.replace(dest)
    size = dest.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f"下載的檔案大小異常({size} bytes),可能是錯誤頁面而非權重檔")
    print(f"\r  完成:{dest}  ({_human_size(size)})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="yolo11n", choices=SIZES,
        help="指定要下載的模型大小(忽略若用 --all)",
    )
    parser.add_argument("--all", action="store_true", help="下載所有大小")
    parser.add_argument(
        "--target", type=Path,
        default=Path(__file__).resolve().parent.parent / "resources" / "models",
    )
    parser.add_argument("--force", action="store_true", help="覆蓋已存在的檔案")
    args = parser.parse_args(argv)

    models = SIZES if args.all else [args.model]
    for name in models:
        filename = f"{name}.pt"
        dest = args.target / filename
        if dest.exists() and not args.force:
            print(f"跳過 {filename}(已存在於 {dest};用 --force 覆蓋)")
            continue
        url = _BASE_URL + filename
        print(f"下載 {filename}")
        print(f"  從 {url}")
        try:
            _download(url, dest)
        except Exception as e:  # noqa: BLE001
            print(f"\n  失敗:{e}", file=sys.stderr)
            return 2
        print(f"  sha256 = {_sha256(dest)[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
