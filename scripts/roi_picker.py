"""ROI + Homography 互動式標註工具。

流程(點擊順序要對應 --world 的世界座標):
  1. 視窗開啟後,依序點擊影像中對應 --world 各座標的地面參考點
  2. Homography 完成後自動進入 ROI 模式:點擊新增斑馬線多邊形頂點
  3. 按 c 關閉多邊形,按 s 存檔

Usage:
    python scripts/roi_picker.py \\
        --video path/to/video.mp4 \\
        --world "(0,0),(4,0),(4,1.2),(0,1.2)" \\
        [--frame 0] [--out-dir .]

世界座標的原點與軸方向由使用者自訂。建議:
  - 原點放在斑馬線靠近路邊的一個端點
  - x 軸沿斑馬線長度方向(公尺)
  - y 軸沿斑馬線寬度方向(公尺)
  - 枕木紋寬度 ≈ 0.4 m,見 docs/legal-rules.md

Keys:
  click  新增目前模式的點
  b      回退(刪最後一個點)
  r      重置目前模式
  c      ROI 模式:關閉多邊形
  s      存檔(需完成 homography 與 ROI)
  q      離開不存檔
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

STATE_HOM = "homography"
STATE_ROI = "roi"

COLOR_HOM = (0, 255, 255)
COLOR_ROI = (0, 255, 0)
COLOR_INFO = (255, 255, 255)
COLOR_HINT = (200, 200, 200)


def _parse_world_points(s: str) -> list[tuple[float, float]]:
    cleaned = s.replace(" ", "").replace("(", "").replace(")", "")
    parts = [p for p in cleaned.split(",") if p]
    if not parts or len(parts) % 2 != 0:
        raise ValueError(f"世界座標解析失敗:{s!r}")
    pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, len(parts), 2)]
    if len(pts) < 4:
        raise ValueError("至少需要 4 組世界座標點")
    return pts


def _extract_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"無法讀取第 {frame_idx} 幀")
        return frame, fps, total
    finally:
        cap.release()


@dataclass
class PickerState:
    world_points: list[tuple[float, float]]
    hom_image_points: list[tuple[float, float]] = field(default_factory=list)
    roi_points: list[tuple[float, float]] = field(default_factory=list)
    roi_closed: bool = False
    mode: str = STATE_HOM


def _on_mouse(event, x, y, flags, param) -> None:  # noqa: ARG001
    state: PickerState = param
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state.mode == STATE_HOM:
        if len(state.hom_image_points) < len(state.world_points):
            state.hom_image_points.append((float(x), float(y)))
            if len(state.hom_image_points) == len(state.world_points):
                state.mode = STATE_ROI
                print("[picker] Homography 完成,切換到 ROI 模式")
    elif state.mode == STATE_ROI and not state.roi_closed:
        state.roi_points.append((float(x), float(y)))


def _draw(frame: np.ndarray, state: PickerState) -> np.ndarray:
    img = frame.copy()

    for i, (x, y) in enumerate(state.hom_image_points):
        cv2.circle(img, (int(x), int(y)), 6, COLOR_HOM, -1)
        label = f"{i + 1}: {state.world_points[i]}"
        cv2.putText(img, label, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HOM, 1, cv2.LINE_AA)

    if state.roi_points:
        pts = np.array(state.roi_points, dtype=np.int32)
        if state.roi_closed and len(pts) >= 3:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], COLOR_ROI)
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
            cv2.polylines(img, [pts], True, COLOR_ROI, 2)
        elif len(pts) >= 2:
            cv2.polylines(img, [pts], False, COLOR_ROI, 2)
        for x, y in state.roi_points:
            cv2.circle(img, (int(x), int(y)), 5, COLOR_ROI, -1)

    hud = []
    if state.mode == STATE_HOM:
        i, n = len(state.hom_image_points), len(state.world_points)
        if i < n:
            hud.append(f"[Homography {i + 1}/{n}] 點擊世界座標 {state.world_points[i]} 對應位置")
        else:
            hud.append(f"[Homography 完成 {n}/{n}]")
    else:
        if state.roi_closed:
            hud.append(f"[ROI 已關閉 · {len(state.roi_points)} 頂點]  按 s 存檔")
        else:
            hud.append(f"[ROI · {len(state.roi_points)} 頂點]  點擊新增,c 關閉,b 回退")
    hud.append("keys: click=add  b=undo  r=reset  c=close ROI  s=save  q=quit")

    for i, line in enumerate(hud):
        y = 26 + i * 24
        color = COLOR_INFO if i == 0 else COLOR_HINT
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
    return img


def _save(out_dir: Path, state: PickerState) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_path = out_dir / "roi.json"
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump({"polygon": [list(p) for p in state.roi_points]},
                  f, ensure_ascii=False, indent=2)
    print(f"[picker] 寫入 {roi_path}")

    hom_path = out_dir / "homography.json"
    with open(hom_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_points": [list(p) for p in state.hom_image_points],
                "world_points_meters": [list(p) for p in state.world_points],
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[picker] 寫入 {hom_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--world", required=True,
        help='世界座標列表,例如: "(0,0),(4,0),(4,1.2),(0,1.2)"(公尺)',
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    try:
        world_pts = _parse_world_points(args.world)
    except ValueError as e:
        print(f"錯誤:{e}", file=sys.stderr)
        return 2

    try:
        frame, fps, total = _extract_frame(args.video, args.frame)
    except Exception as e:  # noqa: BLE001
        print(f"錯誤:{e}", file=sys.stderr)
        return 2

    h, w = frame.shape[:2]
    print(f"[picker] 影片 {w}x{h}, {fps:.2f} fps, {total} frames")
    print(f"[picker] 使用第 {args.frame} 幀")
    print(f"[picker] {len(world_pts)} 組世界座標:{world_pts}")

    state = PickerState(world_points=world_pts)
    win = "ZebraGuard ROI Picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse, state)

    while True:
        cv2.imshow(win, _draw(frame, state))
        key = cv2.waitKey(30) & 0xFF

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("[picker] 視窗關閉,未存檔")
            return 1

        if key == 255:
            continue

        if key == ord("q"):
            print("[picker] 離開,未存檔")
            cv2.destroyAllWindows()
            return 1
        if key == ord("b"):
            if state.mode == STATE_HOM and state.hom_image_points:
                state.hom_image_points.pop()
            elif state.mode == STATE_ROI:
                if state.roi_closed:
                    state.roi_closed = False
                elif state.roi_points:
                    state.roi_points.pop()
        elif key == ord("r"):
            if state.mode == STATE_HOM:
                state.hom_image_points.clear()
            else:
                state.roi_points.clear()
                state.roi_closed = False
        elif key == ord("c"):
            if state.mode == STATE_ROI and not state.roi_closed:
                if len(state.roi_points) >= 3:
                    state.roi_closed = True
                    print(f"[picker] ROI 關閉,{len(state.roi_points)} 頂點")
                else:
                    print(f"[picker] ROI 至少需 3 頂點,目前 {len(state.roi_points)}")
        elif key == ord("s"):
            if len(state.hom_image_points) < len(state.world_points):
                print(f"[picker] Homography 尚未完成 "
                      f"({len(state.hom_image_points)}/{len(state.world_points)})")
                continue
            if not state.roi_closed or len(state.roi_points) < 3:
                print("[picker] ROI 尚未關閉")
                continue
            _save(args.out_dir, state)
            cv2.destroyAllWindows()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
