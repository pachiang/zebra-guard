"""開發模式啟動主程式。

Usage:
    python scripts/dev_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))

    from zebraguard.__main__ import main as app_main  # noqa: WPS433 (延遲匯入是刻意的)
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
