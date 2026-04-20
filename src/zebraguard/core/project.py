"""專案 (.zgproj) 資料模型:資料夾 + project.json + project.db。

一個專案 = 一個資料夾,結構:

    my_crossing.zgproj/
      project.json       metadata(版本、影片、ROI、homography、設定、進度)
      project.db         SQLite(meta、violations、使用者標註)
      thumbnails/        關鍵幀快取(未來)
      exports/           使用者匯出的片段與報告(未來)

對外介面:
  Project.create(path, video_path) → 新建
  Project.load(path)                → 載入既有
  project.save_roi(roi)
  project.save_homography(hom)
  project.save_config(cfg)
  project.save_violations(list[ViolationEvent])   # 覆蓋
  project.load_violations()                       # 含 user_status
  project.update_user_status(vid, "accepted" | "rejected" | "pending", note=...)
  project.close()                                 # 或用 with

本模組**不依賴 PySide6**。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from zebraguard.ml.types import ViolationEvent

PROJECT_FILE_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS violations (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    pedestrian_track_id    INTEGER NOT NULL,
    vehicle_track_id       INTEGER NOT NULL,
    start_sec              REAL    NOT NULL,
    end_sec                REAL    NOT NULL,
    min_distance_m         REAL    NOT NULL,
    min_vehicle_speed_kmh  REAL    NOT NULL,
    closest_frame          INTEGER NOT NULL,
    closest_time_sec       REAL    NOT NULL,
    user_status            TEXT    NOT NULL DEFAULT 'pending'
        CHECK (user_status IN ('pending', 'accepted', 'rejected')),
    user_note              TEXT,
    created_at             TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_violations_start ON violations(start_sec);

-- v7 zero-shot pipeline 輸出。
-- 與 violations 表分開存放:violations 對應舊的 homography 管線(保留以相容 CLI / tests),
-- events 對應現行 v7 preset 的結果,UI 全部走這張表。
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    start_frame     INTEGER NOT NULL,
    end_frame       INTEGER NOT NULL,
    start_sec       REAL    NOT NULL,
    end_sec         REAL    NOT NULL,
    min_distance_px REAL    NOT NULL,
    peak_speed_px   REAL    NOT NULL,
    ped_track_ids   TEXT    NOT NULL DEFAULT '[]',
    veh_track_ids   TEXT    NOT NULL DEFAULT '[]',
    user_status     TEXT    NOT NULL DEFAULT 'pending'
        CHECK (user_status IN ('pending', 'accepted', 'rejected')),
    user_note       TEXT,
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_sec);
"""

_VIDEO_HASH_BYTES = 8 * 1024 * 1024  # 前 8 MB,對 12 小時影片全 hash 太慢


@dataclass
class ProjectMeta:
    """project.json 結構。"""

    version: int = PROJECT_FILE_VERSION
    created_at: str = ""
    updated_at: str = ""
    video_path: str = ""
    video_sha256_partial: str = ""  # 前 8 MB 的 SHA256;為身分識別而非完整校驗
    video_fps: float = 0.0
    video_duration_sec: float = 0.0
    video_width: int = 0
    video_height: int = 0
    video_frame_count: int = 0
    roi: dict[str, Any] = field(default_factory=dict)  # RoiPolygon.to_json()
    homography: dict[str, Any] = field(default_factory=dict)  # Homography.to_json()
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=lambda: {"stage": "created"})


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _compute_partial_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(_VIDEO_HASH_BYTES))
    except OSError:
        return ""
    return h.hexdigest()


class Project:
    """.zgproj 專案封裝。"""

    def __init__(self, path: Path, meta: ProjectMeta, conn: sqlite3.Connection) -> None:
        self.path = path
        self.meta = meta
        self._conn = conn

    # ---- 生命週期 -----------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path, video_path: str | Path) -> Self:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Project 已存在: {path}")
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"影片不存在: {video_path}")

        path.mkdir(parents=True)
        (path / "thumbnails").mkdir()
        (path / "exports").mkdir()

        meta = ProjectMeta(
            created_at=_now_iso(),
            updated_at=_now_iso(),
            video_path=str(video_path),
            video_sha256_partial=_compute_partial_hash(video_path),
        )
        # 試著填入影片維度;失敗不致命,使用者可之後補
        try:
            from zebraguard.ml.detection import probe_video

            info = probe_video(video_path)
            meta.video_fps = info.fps
            meta.video_duration_sec = info.duration_sec
            meta.video_width = info.width
            meta.video_height = info.height
            meta.video_frame_count = info.frame_count
        except Exception:  # noqa: BLE001
            pass

        conn = sqlite3.connect(path / "project.db")
        conn.executescript(_SCHEMA)
        conn.commit()

        project = cls(path, meta, conn)
        project._save_meta()
        return project

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        json_path = path / "project.json"
        db_path = path / "project.db"
        if not json_path.exists():
            raise FileNotFoundError(f"找不到 project.json: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version")
        if version != PROJECT_FILE_VERSION:
            raise ValueError(f"不支援的 project.json 版本: {version}(預期 {PROJECT_FILE_VERSION})")

        # 只取 ProjectMeta 認識的欄位,避免未來新欄位導致 TypeError
        fields = {k: v for k, v in data.items() if k in ProjectMeta.__dataclass_fields__}
        meta = ProjectMeta(**fields)

        conn = sqlite3.connect(db_path)
        conn.executescript(_SCHEMA)  # idempotent,確保表格存在
        return cls(path, meta, conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ---- Meta 操作 -----------------------------------------------------------

    def _save_meta(self) -> None:
        self.meta.updated_at = _now_iso()
        tmp = self.path / "project.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, ensure_ascii=False, indent=2)
        tmp.replace(self.path / "project.json")

    def save_metadata(self) -> None:
        self._save_meta()

    def save_roi(self, roi_json: dict[str, Any]) -> None:
        self.meta.roi = roi_json
        self._save_meta()

    def save_homography(self, hom_json: dict[str, Any]) -> None:
        self.meta.homography = hom_json
        self._save_meta()

    def save_config(self, config_json: dict[str, Any]) -> None:
        self.meta.pipeline_config = config_json
        self._save_meta()

    def update_progress(self, stage: str, **extra: Any) -> None:
        self.meta.progress = {"stage": stage, **extra}
        self._save_meta()

    # ---- Violations ---------------------------------------------------------

    def save_violations(self, violations: list[ViolationEvent]) -> None:
        """覆蓋儲存違規清單。既有 user_status 會被清掉——本函式用於一次性寫入
        pipeline 結果。之後的標註用 update_user_status。"""
        with self._conn:
            self._conn.execute("DELETE FROM violations")
            self._conn.executemany(
                """INSERT INTO violations
                   (pedestrian_track_id, vehicle_track_id, start_sec, end_sec,
                    min_distance_m, min_vehicle_speed_kmh, closest_frame, closest_time_sec)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        v.pedestrian_track_id,
                        v.vehicle_track_id,
                        v.start_sec,
                        v.end_sec,
                        v.min_distance_m,
                        v.min_vehicle_speed_kmh,
                        v.closest_frame,
                        v.closest_time_sec,
                    )
                    for v in violations
                ],
            )

    def load_violations(self) -> list[dict[str, Any]]:
        """含 id、user_status、user_note,供 Review UI 使用。"""
        cur = self._conn.execute(
            """SELECT id, pedestrian_track_id, vehicle_track_id, start_sec, end_sec,
                      min_distance_m, min_vehicle_speed_kmh, closest_frame, closest_time_sec,
                      user_status, user_note
               FROM violations ORDER BY start_sec, id"""
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def update_user_status(
        self, violation_id: int, status: str, note: str | None = None
    ) -> None:
        if status not in ("pending", "accepted", "rejected"):
            raise ValueError(f"無效 status: {status}")
        with self._conn:
            cur = self._conn.execute(
                "UPDATE violations SET user_status = ?, user_note = ? WHERE id = ?",
                (status, note, violation_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"找不到 violation id={violation_id}")

    # ---- 雜項 ---------------------------------------------------------------

    @property
    def accepted_violations(self) -> list[dict[str, Any]]:
        return [v for v in self.load_violations() if v["user_status"] == "accepted"]

    # ---- Events (v7 zero-shot pipeline) ------------------------------------

    def save_events(self, events: list[dict[str, Any]]) -> None:
        """覆蓋儲存 v7 pipeline 的事件清單。

        `events` 的每個元素需含:start_frame / end_frame / start_sec / end_sec /
        min_distance_px / peak_speed_px / ped_track_ids / veh_track_ids。
        """
        with self._conn:
            self._conn.execute("DELETE FROM events")
            self._conn.executemany(
                """INSERT INTO events
                   (start_frame, end_frame, start_sec, end_sec,
                    min_distance_px, peak_speed_px, ped_track_ids, veh_track_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        int(e["start_frame"]),
                        int(e["end_frame"]),
                        float(e["start_sec"]),
                        float(e["end_sec"]),
                        float(e.get("min_distance_px", 0.0)),
                        float(e.get("peak_speed_px", 0.0)),
                        json.dumps(list(e.get("ped_track_ids", []))),
                        json.dumps(list(e.get("veh_track_ids", []))),
                    )
                    for e in events
                ],
            )

    def load_events(self) -> list[dict[str, Any]]:
        """含 id、user_status、user_note;ped/veh_track_ids 已 JSON parsed。"""
        cur = self._conn.execute(
            """SELECT id, start_frame, end_frame, start_sec, end_sec,
                      min_distance_px, peak_speed_px, ped_track_ids, veh_track_ids,
                      user_status, user_note
               FROM events ORDER BY start_sec, id"""
        )
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
        for r in rows:
            r["ped_track_ids"] = json.loads(r["ped_track_ids"] or "[]")
            r["veh_track_ids"] = json.loads(r["veh_track_ids"] or "[]")
        return rows

    def update_event_status(
        self, event_id: int, status: str, note: str | None = None
    ) -> None:
        if status not in ("pending", "accepted", "rejected"):
            raise ValueError(f"無效 status: {status}")
        with self._conn:
            cur = self._conn.execute(
                "UPDATE events SET user_status = ?, user_note = ? WHERE id = ?",
                (status, note, event_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"找不到 event id={event_id}")

    def update_event_range(
        self, event_id: int, start_sec: float, end_sec: float, fps: float
    ) -> None:
        """使用者於 Review Deck 微調事件起訖時段。frame 欄位同步由 fps 換算。"""
        if end_sec <= start_sec:
            raise ValueError("end_sec 必須大於 start_sec")
        with self._conn:
            cur = self._conn.execute(
                """UPDATE events
                   SET start_sec = ?, end_sec = ?, start_frame = ?, end_frame = ?
                   WHERE id = ?""",
                (
                    float(start_sec),
                    float(end_sec),
                    int(round(start_sec * fps)),
                    int(round(end_sec * fps)),
                    event_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"找不到 event id={event_id}")

    @property
    def accepted_events(self) -> list[dict[str, Any]]:
        return [e for e in self.load_events() if e["user_status"] == "accepted"]
