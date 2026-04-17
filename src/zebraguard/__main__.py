"""Entry point: `python -m zebraguard` 或 PyInstaller 產出的 exe 會跑這裡。"""

from __future__ import annotations


def main() -> int:
    # TODO: 初始化 QApplication、設定、log,開主視窗
    raise NotImplementedError("UI 尚未實作;見 docs/mvp.md")


if __name__ == "__main__":
    raise SystemExit(main())
