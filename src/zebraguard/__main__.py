"""Entry point: `python -m zebraguard`。"""

from __future__ import annotations

from zebraguard.ui.app import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
