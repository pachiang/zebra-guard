"""Project (.zgproj) 生命週期測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from zebraguard.core.project import Project, ProjectMeta
from zebraguard.ml.types import ViolationEvent


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """寫一個小檔案冒充影片(Project.create 只會看存在 + 讀前 8 MB hash)。"""
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"\x00" * 64)
    return video


def test_create_project_layout(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        assert (path / "project.json").is_file()
        assert (path / "project.db").is_file()
        assert (path / "thumbnails").is_dir()
        assert (path / "exports").is_dir()
        assert proj.meta.video_path == str(fake_video.resolve())
        assert proj.meta.video_sha256_partial  # non-empty


def test_create_rejects_existing_path(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    path.mkdir()
    with pytest.raises(FileExistsError):
        Project.create(path, fake_video)


def test_load_round_trip(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as p1:
        p1.save_roi({"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]})
        p1.save_config({"yield_distance_m": 3.0})
        p1.update_progress("analyzing", percent=42)

    with Project.load(path) as p2:
        assert p2.meta.roi == {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        assert p2.meta.pipeline_config == {"yield_distance_m": 3.0}
        assert p2.meta.progress == {"stage": "analyzing", "percent": 42}


def test_load_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Project.load(tmp_path / "nonexistent")


def test_load_rejects_wrong_version(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    Project.create(path, fake_video).close()
    # 竄改版本號
    json_path = path / "project.json"
    data = json_path.read_text(encoding="utf-8")
    json_path.write_text(data.replace('"version": 1', '"version": 99'), encoding="utf-8")
    with pytest.raises(ValueError, match="不支援"):
        Project.load(path)


def test_load_ignores_unknown_json_fields(tmp_path: Path, fake_video: Path) -> None:
    """未來版本如果加欄位,舊 code load 不應該 TypeError。"""
    import json

    path = tmp_path / "proj.zgproj"
    Project.create(path, fake_video).close()
    json_path = path / "project.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["some_future_field"] = "ignored"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    Project.load(path).close()  # should not raise


def _fake_violation(start: float, end: float, ped: int = 1, veh: int = 2) -> ViolationEvent:
    return ViolationEvent(
        pedestrian_track_id=ped,
        vehicle_track_id=veh,
        start_sec=start,
        end_sec=end,
        min_distance_m=1.5,
        min_vehicle_speed_kmh=30.0,
        closest_frame=int(start * 30),
        closest_time_sec=start + 0.1,
    )


def test_save_and_load_violations(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        proj.save_violations([
            _fake_violation(10.0, 12.0, ped=1, veh=2),
            _fake_violation(25.0, 27.0, ped=3, veh=4),
        ])
        rows = proj.load_violations()
        assert len(rows) == 2
        assert rows[0]["start_sec"] == 10.0
        assert rows[0]["user_status"] == "pending"
        assert rows[1]["vehicle_track_id"] == 4


def test_save_violations_replaces_existing(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        proj.save_violations([_fake_violation(1.0, 2.0)])
        proj.save_violations([_fake_violation(5.0, 6.0), _fake_violation(7.0, 8.0)])
        rows = proj.load_violations()
        assert [r["start_sec"] for r in rows] == [5.0, 7.0]


def test_update_user_status(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        proj.save_violations([_fake_violation(10.0, 12.0), _fake_violation(20.0, 22.0)])
        rows = proj.load_violations()
        proj.update_user_status(rows[0]["id"], "accepted", note="車牌 ABC-1234")
        proj.update_user_status(rows[1]["id"], "rejected")
        rows = proj.load_violations()
        statuses = {r["id"]: r["user_status"] for r in rows}
        assert statuses[rows[0]["id"]] == "accepted"
        assert statuses[rows[1]["id"]] == "rejected"
        # accepted_violations 只回採用的
        assert len(proj.accepted_violations) == 1


def test_update_user_status_rejects_bad_status(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        proj.save_violations([_fake_violation(1.0, 2.0)])
        vid = proj.load_violations()[0]["id"]
        with pytest.raises(ValueError, match="無效 status"):
            proj.update_user_status(vid, "maybe")


def test_update_user_status_rejects_missing(tmp_path: Path, fake_video: Path) -> None:
    path = tmp_path / "proj.zgproj"
    with Project.create(path, fake_video) as proj:
        with pytest.raises(KeyError):
            proj.update_user_status(99999, "accepted")


def test_project_meta_defaults() -> None:
    meta = ProjectMeta()
    assert meta.version == 1
    assert meta.progress == {"stage": "created"}
