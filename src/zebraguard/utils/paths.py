"""專案根目錄與常見資源路徑。

開發時跑 `python -m zebraguard`,PyInstaller 打包後以 `sys._MEIPASS` 為根。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """回傳 zebra-guard repo 根目錄(或 PyInstaller 解包後的 _MEIPASS)。"""
    # PyInstaller --onefile 解包後 sys._MEIPASS 會指向臨時資源根
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    # Dev 模式:此檔在 src/zebraguard/utils/paths.py,往上三層是 repo 根
    return Path(__file__).resolve().parent.parent.parent.parent


def scripts_dir() -> Path:
    return project_root() / "scripts"


def presets_dir() -> Path:
    return scripts_dir() / "presets"


def resources_dir() -> Path:
    return project_root() / "resources"


def ffmpeg_binary() -> Path | None:
    """回傳綁附的 ffmpeg 路徑(若未綁附則 None,呼叫端應 fallback 到 PATH)。"""
    ff = resources_dir() / "ffmpeg" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    return ff if ff.is_file() else None


def user_data_dir() -> Path:
    """%APPDATA%\\ZebraGuard or ~/.local/share/ZebraGuard 等。"""
    if sys.platform == "win32":
        import os
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "ZebraGuard"
    return Path.home() / ".local" / "share" / "ZebraGuard"


def user_settings_file() -> Path:
    return user_data_dir() / "settings.ini"
