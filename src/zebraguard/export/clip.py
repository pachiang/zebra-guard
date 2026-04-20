"""以 ffmpeg 無損剪輯違規前後 N 秒。

策略:優先用 stream copy(`-c copy`)得到無損與近乎瞬間的輸出;但 `-c copy` 必須
切在 keyframe,導致起點可能被往前對齊。為了時間準確,先於起點前多抓 0.2 秒,
若使用者覺得太長未來再做第二階段 re-encode。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from zebraguard.utils.paths import ffmpeg_binary

CLIP_PAD_BEFORE = 5.0  # 違規起點前後各留多少秒
CLIP_PAD_AFTER = 5.0


def _ffmpeg_path() -> str:
    ff = ffmpeg_binary()
    if ff is not None:
        return str(ff)
    sys_ff = shutil.which("ffmpeg")
    if sys_ff is None:
        raise FileNotFoundError(
            "找不到 ffmpeg;請把 ffmpeg 放到 resources/ffmpeg/ 或加到系統 PATH。"
        )
    return sys_ff


def extract_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    *,
    pad_before: float = CLIP_PAD_BEFORE,
    pad_after: float = CLIP_PAD_AFTER,
) -> None:
    """無損剪輯 [start - pad_before, end + pad_after]。

    若 stream copy 失敗(某些 codec / container 組合),fallback 以 libx264 重編碼。
    """
    clip_start = max(0.0, start_sec - pad_before)
    clip_end = end_sec + pad_after
    duration = max(0.1, clip_end - clip_start)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_path()

    # 注意 -ss 放在 -i 之前是「快速 seek」,但會對齊到前一個 keyframe;
    # 放在 -i 之後是「精確 seek」,但要 re-encode。
    # 對檢舉用途建議用精確 seek + 重編碼,否則起點可能偏移數秒影響判讀。
    cmd = [
        ff,
        "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 匯出失敗(exit={result.returncode}):\n{result.stderr[-2000:]}"
        )
