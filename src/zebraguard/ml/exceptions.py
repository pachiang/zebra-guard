"""Pipeline 共用例外型別。"""

from __future__ import annotations


class Cancelled(Exception):
    """當 cancel_event 在 pipeline 迴圈中被 set 時丟出。

    共用給 dashcam(`scripts/zero_shot_detect.py`)與 static parking
    (`scripts/static_parking_detect.py`);worker 只要 catch 這一個型別即可。
    """
