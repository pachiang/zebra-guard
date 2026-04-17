"""斑馬線(枕木紋)自動偵測 + 自動產生 ROI 與 homography。

演算法(純 CV,不需訓練):
  1. Adaptive threshold 找白色區域
  2. Contour filter 保留「細長矩形」(minAreaRect aspect ratio in [2.5, 25])
  3. 依長軸方向分群(tolerance ±10°)
  4. 群組內排序(沿條紋法線方向),用「等距週期」score 挑最像斑馬線的群
  5. 套用**台灣標準**:枕木紋寬 0.4 m、週期(條紋 + 間隔)1.0 m
  6. 群組的 oriented bounding rect 四角 → ROI
  7. 由條紋數量 × 週期(跨越方向,精確)+ 條紋長度 × 像素比例(沿條紋方向,估計)
     組 homography

Taiwan constants source: 道路交通標誌標線號誌設置規則
  - 枕木紋寬度 40 cm
  - 枕木紋間隔 60 cm(部分資料標示 40 cm,實作時可由參數覆蓋)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 台灣標準(見 docs/legal-rules.md,亦可由參數覆寫)
TW_STRIPE_WIDTH_M = 0.4
TW_STRIPE_PERIOD_M = 1.0  # 條紋寬 + 間隔
# 預設斑馬線「行走方向長度」=「單條條紋的長軸長度」
# 台灣一般斑馬線橫跨車道數,單車道 3 m、雙向 4-6 m;預設保守取 3.5 m
DEFAULT_CROSSING_LENGTH_M = 3.5

# 偵測參數
_MIN_STRIPE_AREA_PX = 800
_MIN_ASPECT = 2.5
_MAX_ASPECT = 12.0
_MIN_SHORT_SIDE_PX = 5
# 條紋最大長度(相對於畫面較短邊的比例)——避免路邊連續白線被當成條紋
_MAX_LONG_SIDE_RATIO = 0.5
_ORIENTATION_TOL_DEG = 10.0
_MIN_STRIPES_IN_GROUP = 4
# Top-hat kernel 約等於預期條紋寬度 * 3
_TOPHAT_KERNEL_PX = 51


@dataclass(slots=True)
class Stripe:
    center: tuple[float, float]
    angle_deg: float  # 長軸方向 [0, 180)
    long_px: float
    short_px: float
    corners: np.ndarray = field(repr=False)  # shape (4, 2)


@dataclass(slots=True)
class DetectedCrosswalk:
    stripes: list[Stripe]
    roi_image_corners: np.ndarray  # (4, 2) — 首尾條紋外側端點組成的四邊形
    image_points: list[list[float]]  # 2N 點(N 條條紋 × 每條 2 端點)
    world_points_meters: list[list[float]]
    crossing_length_m: float  # 行走方向長度(沿條紋長軸)
    crossing_width_m: float   # 跨越方向長度(由條紋數 × 週期推出)
    avg_thickness_px: float


def _normalise_angle(a: float) -> float:
    """把角度壓到 [0, 180)。"""
    a = a % 180.0
    if a < 0:
        a += 180.0
    return a


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def binarise(frame: np.ndarray) -> np.ndarray:
    """取白色條紋。

    用 **top-hat transform**(灰階形態學):
      top_hat = gray - opening(gray, kernel)
    留下比周圍亮且比 kernel 小的結構(= 條紋),去掉大面積亮度(= 路面、天空)。
    這對光影不均的戶外畫面特別有效。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    k = _TOPHAT_KERNEL_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    # 小缺口補起來 + 小噪點去掉
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    return binary


def find_stripe_candidates(binary: np.ndarray) -> list[Stripe]:
    h_img, w_img = binary.shape[:2]
    max_long_px = min(h_img, w_img) * _MAX_LONG_SIDE_RATIO
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Stripe] = []
    for contour in contours:
        if cv2.contourArea(contour) < _MIN_STRIPE_AREA_PX:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        long_side = max(w, h)
        short_side = min(w, h)
        if short_side < _MIN_SHORT_SIDE_PX:
            continue
        if long_side > max_long_px:
            continue  # 過長 → 可能是路邊線/欄杆
        aspect = long_side / max(short_side, 1e-6)
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            continue
        long_axis_angle = angle if w >= h else (angle + 90.0)
        out.append(Stripe(
            center=(float(cx), float(cy)),
            angle_deg=_normalise_angle(long_axis_angle),
            long_px=float(long_side),
            short_px=float(short_side),
            corners=cv2.boxPoints(rect),
        ))
    return out


def group_parallel(stripes: list[Stripe]) -> list[list[Stripe]]:
    """依長軸方向分群,代表角度用群組平均。"""
    groups: list[list[Stripe]] = []
    for s in stripes:
        placed = False
        for g in groups:
            rep = float(np.mean([x.angle_deg for x in g]))
            if _angle_diff(s.angle_deg, rep) < _ORIENTATION_TOL_DEG:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
    return [g for g in groups if len(g) >= _MIN_STRIPES_IN_GROUP]


def _project_centers_along_normal(group: list[Stripe]) -> tuple[np.ndarray, np.ndarray]:
    """把群組內每個條紋中心投影到「垂直條紋方向」上並排序。

    回傳:(排序後的投影值, 排序後的索引)
    """
    angle = float(np.mean([s.angle_deg for s in group]))
    angle_rad = np.deg2rad(angle)
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    centers = np.array([s.center for s in group])
    proj = centers @ normal
    order = np.argsort(proj)
    return proj[order], order


def score_group(group: list[Stripe]) -> float:
    """高分 ≈ 像斑馬線。

    權重:
      - count 佔 50% — 斑馬線條紋多;少量雜訊容易剛好等距
      - uniformity(等距性)佔 30%
      - size spread(條紋長度相近性)佔 20% — 斑馬線的條紋長度約略一致
    """
    if len(group) < _MIN_STRIPES_IN_GROUP:
        return -1.0
    proj_sorted, _ = _project_centers_along_normal(group)
    gaps = np.diff(proj_sorted)
    if len(gaps) == 0 or gaps.mean() <= 0:
        return -1.0
    cv_gap = gaps.std() / (gaps.mean() + 1e-6)
    uniformity = max(0.0, 1.0 - cv_gap)
    count = min(1.0, len(group) / 8.0)
    long_lens = np.array([s.long_px for s in group])
    cv_len = long_lens.std() / (long_lens.mean() + 1e-6)
    size_consistency = max(0.0, 1.0 - cv_len)
    return count * 0.5 + uniformity * 0.3 + size_consistency * 0.2


def pick_best_group(groups: list[list[Stripe]]) -> list[Stripe] | None:
    if not groups:
        return None
    best, best_score = None, -1.0
    for g in groups:
        s = score_group(g)
        if s > best_score:
            best_score = s
            best = g
    return best if best_score > 0.3 else None


def _order_corners_tlbr(corners: np.ndarray) -> np.ndarray:
    """把 minAreaRect 的 4 個角排成 TL, TR, BR, BL。

    以中心為參考,依極座標角度分四象限。
    """
    center = corners.mean(axis=0)
    # 角度 0 從 +x,逆時針
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    # 分區:
    #   TL: x < cx AND y < cy
    #   TR: x > cx AND y < cy
    #   BR: x > cx AND y > cy
    #   BL: x < cx AND y > cy
    out = np.zeros((4, 2), dtype=np.float64)
    assigned = [False] * 4
    for pt in corners:
        x, y = pt
        if x <= center[0] and y <= center[1]:
            idx = 0  # TL
        elif x > center[0] and y <= center[1]:
            idx = 1  # TR
        elif x > center[0] and y > center[1]:
            idx = 2  # BR
        else:
            idx = 3  # BL
        if not assigned[idx]:
            out[idx] = pt
            assigned[idx] = True
        else:
            # 衝突:退回用排序方式
            break
    else:
        return out
    _ = angles  # 沉默 lint
    # Fallback:sum / diff 排序
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmax(d)]
    bl = corners[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)


def _stripe_endpoints(stripe: Stripe) -> tuple[np.ndarray, np.ndarray]:
    """取出條紋長軸兩端的端點中心。

    條紋 4 個角落沿著長軸方向投影排序,前 2 個平均 = 左端;後 2 個平均 = 右端。
    回傳 (left_end, right_end) 各為 shape (2,) 的影像座標點。
    """
    angle_rad = np.deg2rad(stripe.angle_deg)
    long_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    proj = stripe.corners @ long_dir
    order = np.argsort(proj)
    sorted_corners = stripe.corners[order]
    left_end = sorted_corners[:2].mean(axis=0)
    right_end = sorted_corners[2:].mean(axis=0)
    return left_end, right_end


def detect(
    frame: np.ndarray,
    *,
    stripe_width_m: float = TW_STRIPE_WIDTH_M,
    stripe_period_m: float = TW_STRIPE_PERIOD_M,
    crossing_length_m: float = DEFAULT_CROSSING_LENGTH_M,
) -> DetectedCrosswalk | None:
    """主入口:回傳 DetectedCrosswalk 或 None。

    Homography 由 **每一條條紋的 2 個端點** 組成 2N 組對應點,
    透過 cv2.findHomography 解出。這讓透視扭曲被自動處理,不需要假設均勻尺度。

    未知量 `crossing_length_m`(斑馬線的長度,也就是行人走的距離)
    由參數指定,預設 3.5 m(台灣單車道約 3 m、雙向 4-6 m)。
    """
    binary = binarise(frame)
    stripes = find_stripe_candidates(binary)
    groups = group_parallel(stripes)
    best = pick_best_group(groups)
    if best is None:
        return None

    # 沿法線方向排序條紋
    _, order = _project_centers_along_normal(best)
    sorted_stripes = [best[i] for i in order]
    n_stripes = len(sorted_stripes)

    # 為每條條紋算出世界 Y(跨越方向的累積位置)——
    # 條紋中心 Y = 前一個週期末 + stripe_width/2
    # 公式化:stripe N (from 0) 中心 y = N * period + stripe_width / 2
    # 為了對稱,我們把首條條紋的「遠端邊」當成 y=0
    # 那麼 stripe N 的左右端點都在 y = N * period + stripe_width/2
    # 條紋 0..N-1 的中心 y: 0.5*stripe_width, period+0.5*stripe_width, ...
    # 最後一條中心 y = (N-1)*period + 0.5*stripe_width
    # 整個斑馬線的跨越方向長度 = (N-1)*period + stripe_width
    crossing_width_m = (n_stripes - 1) * stripe_period_m + stripe_width_m

    # 決定哪一端是「左」(x=0),哪一端是「右」(x=crossing_length_m)
    # 所有條紋的 "left_end" 向量大致要一致(不能有些在西邊有些在東邊)
    # 用第一條作為基準,後面每條對齊
    first_left, first_right = _stripe_endpoints(sorted_stripes[0])
    ref_dir = first_right - first_left  # 從左指向右的方向(像素)

    image_pts: list[list[float]] = []
    world_pts: list[list[float]] = []
    # 同時記錄首條和末條的外端點,之後當 ROI 4 角
    first_left_world = first_right_world = None
    last_left_img = last_right_img = None
    last_left_world = last_right_world = None

    for n, stripe in enumerate(sorted_stripes):
        le, re = _stripe_endpoints(stripe)
        # 看 le→re 方向跟 ref 是否同向,若相反則交換
        if np.dot(re - le, ref_dir) < 0:
            le, re = re, le
        wy = n * stripe_period_m + stripe_width_m / 2.0
        image_pts.append([float(le[0]), float(le[1])])
        world_pts.append([0.0, float(wy)])
        image_pts.append([float(re[0]), float(re[1])])
        world_pts.append([float(crossing_length_m), float(wy)])
        if n == 0:
            first_left_world = (0.0, 0.0)
            first_right_world = (crossing_length_m, 0.0)
            first_left_img = le
            first_right_img = re
        if n == n_stripes - 1:
            last_left_img = le
            last_right_img = re
            last_left_world = (0.0, crossing_width_m)
            last_right_world = (crossing_length_m, crossing_width_m)

    assert last_left_img is not None and last_right_img is not None
    assert first_left_world is not None and last_left_world is not None

    # ROI 四個角:首條條紋外側 + 末條條紋外側
    roi_corners = np.array([
        first_left_img,
        first_right_img,
        last_right_img,
        last_left_img,
    ], dtype=np.float64)

    avg_thickness_px = float(np.mean([s.short_px for s in best]))

    return DetectedCrosswalk(
        stripes=sorted_stripes,
        roi_image_corners=roi_corners,
        image_points=image_pts,
        world_points_meters=world_pts,
        crossing_length_m=crossing_length_m,
        crossing_width_m=crossing_width_m,
        avg_thickness_px=avg_thickness_px,
    )


def draw_preview(
    frame: np.ndarray,
    stripes: list[Stripe],
    result: DetectedCrosswalk | None,
) -> np.ndarray:
    out = frame.copy()
    # 所有候選條紋(黃色細線)
    for s in stripes:
        cv2.drawContours(out, [s.corners.astype(np.int32)], 0, (0, 200, 255), 1)
    if result is None:
        cv2.putText(out, "NO CROSSWALK DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    # 被選中的條紋(橘色粗線)
    for s in result.stripes:
        cv2.drawContours(out, [s.corners.astype(np.int32)], 0, (0, 140, 255), 2)
    # ROI(綠色填充半透明 + 粗邊)
    pts = result.roi_image_corners.astype(np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    cv2.polylines(out, [pts], True, (0, 255, 0), 3)
    for i, (x, y) in enumerate(pts):
        label = f"P{i + 1}: {tuple(round(v, 2) for v in result.world_points_meters[i])}m"
        cv2.circle(out, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(out, label, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    info = (
        f"stripes={len(result.stripes)}  "
        f"thickness={result.avg_thickness_px:.1f}px  "
        f"length={result.crossing_length_m:.2f}m  "
        f"width={result.crossing_width_m:.2f}m"
    )
    cv2.putText(out, info, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, info, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
