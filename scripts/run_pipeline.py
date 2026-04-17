"""Dev shim:不安裝套件也能跑 CLI。

正式使用請:`uv run zebraguard-pipeline ...`(等效)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zebraguard.cli import run_pipeline_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_pipeline_main())
