"""Burn event markers from zero_shot_detect.py output into a video.

Reads events.json + source video, writes an annotated mp4 showing:
  - Red banner when an event is active: "EVENT #N   t=1.23s–2.34s   N_peds, N_vehs"
  - Pulsing red circle top-right while active
  - Timeline strip at bottom with all events (red bars), playhead, and scale
  - Countdown label showing time until the next event when idle

Can annotate the full video, or only clips around each event (-/+ pad seconds).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Event:
    idx: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    n_peds: int
    n_vehs: int


def load_events(path: Path) -> tuple[list[Event], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    evs: list[Event] = []
    for i, e in enumerate(data["events"]):
        evs.append(Event(
            idx=i + 1,
            start_frame=e["start_frame"],
            end_frame=e["end_frame"],
            start_sec=e["start_sec"],
            end_sec=e["end_sec"],
            n_peds=len(e["ped_track_ids"]),
            n_vehs=len(e["veh_track_ids"]),
        ))
    return evs, data


def active_event(events: list[Event], frame_idx: int) -> Event | None:
    for e in events:
        if e.start_frame <= frame_idx <= e.end_frame:
            return e
    return None


def next_event(events: list[Event], frame_idx: int) -> Event | None:
    for e in events:
        if e.start_frame > frame_idx:
            return e
    return None


def draw_event_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    active: Event | None,
    next_e: Event | None,
    events: list[Event],
    video_frame_count: int,
) -> np.ndarray:
    H, W = frame.shape[:2]
    out = frame.copy()
    t_cur = frame_idx / fps

    # --- Bottom timeline strip ---
    strip_h = 38
    strip_y0 = H - strip_h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, strip_y0), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    bar_y = strip_y0 + 22
    bar_h = 8
    x_left = 12
    x_right = W - 12
    # baseline
    cv2.line(out, (x_left, bar_y + bar_h // 2), (x_right, bar_y + bar_h // 2),
             (120, 120, 120), 1, cv2.LINE_AA)

    def x_of_frame(fi: int) -> int:
        if video_frame_count <= 1:
            return x_left
        return int(x_left + (x_right - x_left) * fi / (video_frame_count - 1))

    # event bars
    for e in events:
        x1 = x_of_frame(e.start_frame)
        x2 = x_of_frame(e.end_frame)
        if x2 - x1 < 2:
            x2 = x1 + 2
        cv2.rectangle(out, (x1, bar_y), (x2, bar_y + bar_h), (40, 40, 200), -1)
    # playhead
    px = x_of_frame(frame_idx)
    cv2.line(out, (px, bar_y - 6), (px, bar_y + bar_h + 6), (0, 255, 255), 2, cv2.LINE_AA)

    # time tick labels
    total_sec = video_frame_count / fps
    cv2.putText(out, _fmt_time(t_cur), (x_left, strip_y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    end_label = _fmt_time(total_sec)
    (tw, _), _ = cv2.getTextSize(end_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(out, end_label, (x_right - tw, strip_y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    # total event count centered
    summary = f"{len(events)} events flagged"
    (tw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(out, summary, ((W - tw) // 2, strip_y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Top banner ---
    banner_h = 56
    if active is not None:
        pulse = 0.6 + 0.4 * abs(math.sin(frame_idx * 0.25))
        color = (int(40 * pulse), int(40 * pulse), int(255 * pulse))
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, 0), (W, banner_h), color, -1)
        out = cv2.addWeighted(overlay2, 0.80, out, 0.20, 0)

        title = f"EVENT #{active.idx}  YIELDING SUSPECTED"
        cv2.putText(out, title, (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        sub = (f"t = {active.start_sec:.2f}s – {active.end_sec:.2f}s   "
               f"({active.end_sec - active.start_sec:.2f}s)   "
               f"peds={active.n_peds}  vehs={active.n_vehs}")
        cv2.putText(out, sub, (14, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # progress bar within this event
        prog_y = banner_h - 4
        prog_x0 = 12
        prog_x1 = W - 12
        cv2.rectangle(out, (prog_x0, prog_y), (prog_x1, prog_y + 3), (80, 80, 80), -1)
        dur = max(1, active.end_frame - active.start_frame)
        pfrac = (frame_idx - active.start_frame) / dur
        pfrac = min(max(pfrac, 0.0), 1.0)
        cv2.rectangle(out, (prog_x0, prog_y),
                      (prog_x0 + int((prog_x1 - prog_x0) * pfrac), prog_y + 3),
                      (255, 255, 255), -1)

        # pulsing dot top-right
        r = int(8 + 2 * math.sin(frame_idx * 0.25))
        cv2.circle(out, (W - 22, 22), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (W - 22, 22), r, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        # idle: thin banner showing countdown to next event
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, 0), (W, 26), (30, 30, 30), -1)
        out = cv2.addWeighted(overlay2, 0.6, out, 0.4, 0)
        if next_e is not None:
            gap = next_e.start_sec - t_cur
            label = (f"next EVENT #{next_e.idx} in {gap:5.1f}s  "
                     f"(t={next_e.start_sec:.1f}s)")
        else:
            label = f"{len(events)} events total — no more ahead"
        cv2.putText(out, label, (14, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def run(
    events_json: Path,
    video_path: Path,
    output_path: Path,
    *,
    clips_only: bool,
    pad_sec: float,
    max_seconds: float | None,
    codec: str,
) -> None:
    events, meta = load_events(events_json)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {W}x{H} @ {fps:.2f}fps {total} frames ({total/fps:.1f}s)")
    print(f"[events] {len(events)} events from {events_json.name}")

    # Which frames to include
    wanted_ranges: list[tuple[int, int]] = []
    if clips_only:
        pad_f = int(pad_sec * fps)
        for e in events:
            s = max(0, e.start_frame - pad_f)
            t = min(total - 1, e.end_frame + pad_f)
            wanted_ranges.append((s, t))
        # merge overlapping
        wanted_ranges.sort()
        merged: list[tuple[int, int]] = []
        for s, t in wanted_ranges:
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t))
            else:
                merged.append((s, t))
        wanted_ranges = merged
    else:
        wanted_ranges = [(0, total - 1)]

    if max_seconds is not None:
        cap_frame = int(max_seconds * fps)
        wanted_ranges = [(s, min(t, cap_frame)) for s, t in wanted_ranges if s <= cap_frame]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}")

    total_to_write = sum(t - s + 1 for s, t in wanted_ranges)
    print(f"[annotate] writing {len(wanted_ranges)} range(s), {total_to_write} frames")

    written = 0
    prev_range_end: int | None = None
    for s, t in wanted_ranges:
        # Only seek when gap > 60 frames; otherwise keep reading linearly
        if prev_range_end is None or s - prev_range_end > 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            cur = s
        else:
            # skip forward
            cur = (prev_range_end + 1) if prev_range_end is not None else 0
            while cur < s:
                ok = cap.grab()
                if not ok:
                    break
                cur += 1

        while cur <= t:
            ok, frame = cap.read()
            if not ok:
                break
            active = active_event(events, cur)
            nxt = next_event(events, cur)
            annotated = draw_event_overlay(frame, cur, fps, active, nxt, events, total)
            writer.write(annotated)
            cur += 1
            written += 1
            if written % 300 == 0:
                print(f"[annotate] {written}/{total_to_write} ({100*written/total_to_write:.1f}%)")
        prev_range_end = t

    cap.release()
    writer.release()
    print(f"[done] wrote {written} frames → {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clips-only", action="store_true",
                   help="Write only clips around each event (±pad-sec), not the full video.")
    p.add_argument("--pad-sec", type=float, default=3.0,
                   help="When --clips-only, include this many seconds before/after each event.")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Cap output duration at this many seconds of source time.")
    p.add_argument("--codec", default="mp4v",
                   help="FourCC codec for VideoWriter. 'mp4v' is safest on Windows.")
    args = p.parse_args()

    run(args.events, args.video, args.out,
        clips_only=args.clips_only,
        pad_sec=args.pad_sec,
        max_seconds=args.max_seconds,
        codec=args.codec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
