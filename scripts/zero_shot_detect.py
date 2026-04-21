"""Zero-shot vehicle-not-yielding detection on crosswalks.

Pipeline:
  1. Mask2Former (facebook/mask2former-swin-large-mapillary-vistas-semantic)
     → zebra-crossing mask (classes: "Crosswalk - Plain" + "Lane Marking - Crosswalk").
     Masks smaller than --min-mask-area-frac of the frame are rejected.
  2. YOLO11 + ByteTrack → person / car / truck / bus / motorcycle tracks
  3. Frame rule (pure 2D, no homography):
       - pedestrian foot point inside dilated mask
       - vehicle foot point ON the dilated mask (prox-px=0, strict) OR within
         --prox-px of it (loose). Default is strict.
       - vehicle moving (pixel displacement > --moving-px)
  4. Temporal aggregation into events with --merge-gap-sec tolerance.

Outputs:
  - JSON list of candidate events with start/end timestamps
  - Optional annotated video (--annotated-out): mask outline + boxes + event banner
  - Optional raw preview (--preview): first N seconds with per-frame overlays
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# progress_cb(stage: str, current: int, total: int, hits: int) -> None
#   stage ∈ {"loading_mask", "loading_yolo", "analyzing", "annotating", "done"}
#   current/total 的單位由 stage 決定:analyzing 為 frame index;annotating 為已寫入幀
ProgressCallback = Callable[[str, int, int, int], None]


class Cancelled(Exception):
    """Raised internally when cancel_event is set. Caller should catch this."""

ROOT = Path(__file__).resolve().parent.parent

# Mapillary Vistas v1.2 (65 classes) labels covering zebra crossings.
# The swin-large model is trained on v1.2, which has two relevant classes:
#   "Crosswalk - Plain"          = full crosswalk surface (area)
#   "Lane Marking - Crosswalk"   = white stripe markings (枕木紋)
# v2.0 label "marking--discrete--crosswalk-zebra" does NOT exist in this model.
ZEBRA_LABELS = ("Crosswalk - Plain", "Lane Marking - Crosswalk")
# COCO class ids used by ultralytics default yolo11 weights
PERSON_ID = 0
VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


@dataclass
class FrameHit:
    frame_idx: int
    t_sec: float
    ped_track_ids: list[int]
    veh_track_ids: list[int]
    min_vehicle_distance_px: float
    vehicle_speed_px: float


@dataclass
class Event:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    ped_track_ids: list[int] = field(default_factory=list)
    veh_track_ids: list[int] = field(default_factory=list)
    min_distance_px: float = 1e9
    peak_speed_px: float = 0.0


def log(msg: str) -> None:
    print(msg, flush=True)


def load_mask2former(
    model_name: str,
    device: torch.device,
    progress_cb: ProgressCallback | None = None,
):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    log(f"[models] loading {model_name} ...")
    if progress_cb is not None:
        progress_cb("loading_mask", 0, 1, 0)
    t0 = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()
    log(f"[models] mask2former ready in {time.monotonic() - t0:.1f}s on {device}")
    if progress_cb is not None:
        progress_cb("loading_mask", 1, 1, 0)

    id2label = model.config.id2label
    zebra_ids: list[int] = []
    for i, name in id2label.items():
        if name in ZEBRA_LABELS:
            zebra_ids.append(int(i))
    if not zebra_ids:
        raise RuntimeError(
            f"None of {ZEBRA_LABELS} found in model labels "
            f"({len(id2label)} classes). Check model version."
        )
    log(f"[models] zebra-crossing class ids = {zebra_ids} "
        f"({[id2label[i] for i in zebra_ids]})")
    return processor, model, zebra_ids


def segment_zebra(
    frame_bgr: np.ndarray,
    processor,
    model,
    zebra_ids: list[int],
    device: torch.device,
    mask_imgsz: int = 0,
) -> np.ndarray:
    """Return uint8 binary mask (0/255) same size as frame for zebra crossing pixels.

    If `mask_imgsz` > 0, resize the input so the short side matches this many
    pixels before Mask2Former; the output mask is upscaled back with
    nearest-neighbor. Significantly reduces GPU load for large frames with
    minimal quality loss for crosswalk-scale targets.
    """
    H_full, W_full = frame_bgr.shape[:2]
    src = frame_bgr
    if mask_imgsz > 0 and min(H_full, W_full) > mask_imgsz:
        if H_full <= W_full:
            new_h = mask_imgsz
            new_w = int(round(W_full * mask_imgsz / H_full))
        else:
            new_w = mask_imgsz
            new_h = int(round(H_full * mask_imgsz / W_full))
        src = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_size = pil.size[::-1]  # (H_small, W_small)
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
    seg_np = seg.cpu().numpy().astype(np.int32)
    mask_small = np.isin(seg_np, zebra_ids).astype(np.uint8) * 255
    if mask_small.shape != (H_full, W_full):
        mask = cv2.resize(mask_small, (W_full, H_full), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small
    return mask


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform: 0 inside mask, positive = pixels to nearest mask pixel."""
    inv = (mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def foot_point(xyxy: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int(y2)


def _bbox_component(xyxy: np.ndarray, labels: np.ndarray,
                    strip_frac: float = 0.15) -> int:
    """Return the crosswalk component id that best overlaps the bottom strip of
    the bbox, or 0 if no overlap. Uses a bottom strip (instead of a single foot
    point) so that a vehicle occluding its own crosswalk can still be matched
    via its bbox edges touching the visible mask pixels.
    """
    H, W = labels.shape
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    h = y2 - y1
    if h <= 0 or x2 <= x1:
        return 0
    y1_strip = max(0, int(y2 - max(1, strip_frac * h)))
    strip = labels[y1_strip:y2, x1:x2]
    if strip.size == 0:
        return 0
    flat = strip.ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return 0
    # Return the most-common component id in the strip
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` bbox area covered by the intersection with `outer`."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / inner_area)


def is_rider(person_bb: np.ndarray, moto_bb: np.ndarray,
             *,
             contain_thresh: float,
             foot_margin_px: int) -> bool:
    """Return True if the person bbox looks like the rider of this moto/bike bbox.

    Three alternative criteria (ANY triggers "rider"):
      1. Standard containment: >= contain_thresh of person bbox lies inside moto bbox
         (robust when rider bbox fits well inside the bike bbox).
      2. Bottom-half overlap: >= contain_thresh of the person's *lower half*
         (the part from mid-y to y2) lies inside the moto bbox.
         YOLO rider bboxes often extend above the bike — this covers that case.
      3. Foot-point inside expanded moto bbox: the bottom-center of the person
         bbox falls inside the moto bbox expanded by foot_margin_px on each
         side. Anchors on the fact that the rider's feet/seat are on the bike.
    """
    # Criterion 1
    if bbox_containment(person_bb, moto_bb) >= contain_thresh:
        return True
    # Criterion 2: lower half of person
    mid_y = 0.5 * (person_bb[1] + person_bb[3])
    lower = np.array([person_bb[0], mid_y, person_bb[2], person_bb[3]])
    if bbox_containment(lower, moto_bb) >= contain_thresh:
        return True
    # Criterion 3: foot-point inside expanded moto bbox
    fx = 0.5 * (person_bb[0] + person_bb[2])
    fy = person_bb[3]
    m = foot_margin_px
    if (moto_bb[0] - m) <= fx <= (moto_bb[2] + m) and \
       (moto_bb[1] - m) <= fy <= (moto_bb[3] + m):
        return True
    return False


def is_ego_vehicle(xyxy: np.ndarray, W: int, H: int,
                   bottom_px: int = 4, min_width_frac: float = 0.40,
                   min_height_frac: float = 0.25) -> bool:
    """Heuristic: dashcam ego vehicle shows as a wide bbox touching the frame bottom.

    Criteria (all must hold):
      - bbox bottom edge within `bottom_px` of frame bottom
      - bbox width >= `min_width_frac` of frame width (covers hood/cowl)
      - bbox height >= `min_height_frac` of frame height (excludes narrow far cars)
    """
    x1, y1, x2, y2 = xyxy
    if (H - 1 - y2) > bottom_px:
        return False
    if (x2 - x1) < min_width_frac * W:
        return False
    if (y2 - y1) < min_height_frac * H:
        return False
    return True


@dataclass
class FrameRecord:
    """Per-processed-frame cache for the annotation pass."""
    idx: int
    mask_idx: int  # -1 if no mask this frame
    detections: list[dict]  # {cls, tid, xyxy: [x1,y1,x2,y2], hit_ped: bool, hit_veh: bool}


def run(
    video_path: Path,
    *,
    model_name: str,
    yolo_weights: str,
    device_str: str,
    stride: int,
    mask_every: int,
    dilate_px: int,
    prox_px: int,
    moving_px: float,
    conf: float,
    imgsz: int,
    merge_gap_sec: float,
    min_event_frames: int,
    output_json: Path | None,
    preview_video: Path | None,
    preview_max_seconds: float,
    exclude_ego: bool,
    ego_bottom_px: int,
    ego_min_width_frac: float,
    ego_min_height_frac: float,
    max_seconds: float | None,
    min_mask_area_frac: float,
    annotated_out: Path | None,
    rider_contain_thresh: float,
    rider_foot_margin_px: int,
    person_conf: float,
    mask_imgsz: int,
    crosswalk_backend: str = "mask2former",
    yolo_seg_weights: str = "",
    yolo_seg_classes: list[str] | None = None,
    yolo_seg_conf: float = 0.25,
    yolo_seg_imgsz: int = 640,
    yolo_seg_min_component_px: int = 200,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    device = torch.device(device_str)

    if crosswalk_backend not in ("mask2former", "yolo_seg"):
        raise ValueError(
            f"未知的 crosswalk_backend: {crosswalk_backend};"
            f"支援 mask2former / yolo_seg"
        )
    log(f"[backend] crosswalk source = {crosswalk_backend}")

    # === Crosswalk backend 載入 ===
    # 注意 mask2former 路徑維持與舊版 byte-level 等價(regression 基準)。
    # yolo_seg 路徑使用 CrosswalkSource 介面。
    processor = mask_model = zebra_ids = None
    crosswalk_source = None  # yolo_seg 用
    if crosswalk_backend == "mask2former":
        processor, mask_model, zebra_ids = load_mask2former(model_name, device, progress_cb)
    else:
        from zebraguard.ml.crosswalk.yolo_seg import YoloSegConfig, YoloSegSource

        if not yolo_seg_weights:
            raise ValueError(
                "crosswalk_backend=yolo_seg 需要 --yolo-seg-weights 指定 .pt 權重"
            )
        log(f"[models] loading YOLO-seg crosswalk weights: {yolo_seg_weights}")
        if progress_cb is not None:
            progress_cb("loading_mask", 0, 1, 0)
        crosswalk_source = YoloSegSource(
            YoloSegConfig(
                weights=yolo_seg_weights,
                class_names=yolo_seg_classes,
                conf=yolo_seg_conf,
                imgsz=yolo_seg_imgsz,
                dilate_px=dilate_px,
                min_mask_area_frac=min_mask_area_frac,
                min_component_px=yolo_seg_min_component_px,
                infer_every=max(1, mask_every),
                device=device_str if device.type != "cpu" else "cpu",
            )
        )
        # 觸發模型載入(讓 progress 能準確反映)
        crosswalk_source._ensure_loaded()
        if progress_cb is not None:
            progress_cb("loading_mask", 1, 1, 0)

    from ultralytics import YOLO

    log(f"[models] loading YOLO {yolo_weights} ...")
    if progress_cb is not None:
        progress_cb("loading_yolo", 0, 1, 0)
    yolo = YOLO(yolo_weights)
    if progress_cb is not None:
        progress_cb("loading_yolo", 1, 1, 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"[video] {video_path.name}: {W}x{H} @ {fps:.2f}fps {total} frames ({total/fps:.1f}s)")

    # Cap input length if requested
    max_frame = total
    if max_seconds is not None:
        max_frame = min(total, int(max_seconds * fps))
        log(f"[video] --max-seconds={max_seconds} → processing first {max_frame} frames")

    min_mask_area_px = int(min_mask_area_frac * W * H)
    log(f"[video] min mask area = {min_mask_area_frac*100:.2f}% of frame = {min_mask_area_px} px")

    writer = None
    if preview_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_video), fourcc, fps, (W, H))

    # Running state
    cached_mask: np.ndarray | None = None  # cleaned (area-filtered) mask
    cached_dilated: np.ndarray | None = None
    cached_dist: np.ndarray | None = None
    cached_labels: np.ndarray | None = None  # per-pixel component id (0=bg, 1..N)
    cached_area: int = 0  # area of cached_mask, 0 if rejected
    cached_n_components: int = 0
    last_seg_frame = -10**9

    # Cache for the annotation pass (only if --annotated-out set)
    cache_enabled = annotated_out is not None
    frame_records: list[FrameRecord] = []
    mask_cache: list[np.ndarray] = []  # stored masks for annotation; idx referenced by FrameRecord.mask_idx
    cur_mask_idx: int = -1

    prev_centers: dict[int, tuple[float, float]] = {}  # track_id -> (cx, cy)
    frame_hits: list[FrameHit] = []

    t_start = time.monotonic()
    idx = 0

    while True:
        if cancel_event is not None and cancel_event.is_set():
            cap.release()
            if writer is not None:
                writer.release()
            raise Cancelled()
        if idx >= max_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride != 0:
            if writer is not None and idx / fps <= preview_max_seconds:
                writer.write(frame)
            idx += 1
            continue

        # Refresh zebra-crossing mask periodically.
        # 用 cached_labels 做 "first iteration" 判斷,而不是 cached_mask,
        # 因為 yolo_seg backend 可能讓 cached_mask 保持 None。
        if cached_labels is None or idx - last_seg_frame >= mask_every:
            if crosswalk_backend == "mask2former":
                raw = segment_zebra(frame, processor, mask_model, zebra_ids, device,
                                    mask_imgsz=mask_imgsz)
                area = int((raw > 0).sum())
                if area < min_mask_area_px:
                    # Mask too small this cycle → clear. (v6 kept the previous mask
                    # but on moving dashcam this leaves ghost zebras at their old
                    # image positions as the camera drives forward.)
                    cached_mask = np.zeros_like(raw)
                    cached_dilated = np.zeros_like(raw)
                    cached_dist = np.full_like(raw, 10**6, dtype=np.float32)
                    cached_labels = np.zeros(raw.shape, dtype=np.int32)
                    cached_area = 0
                    cached_n_components = 0
                else:
                    cached_mask = raw
                    cached_dilated = dilate_mask(cached_mask, dilate_px)
                    cached_dist = distance_to_mask(cached_dilated)
                    # Connected components on the DILATED mask: nearby crosswalks
                    # that merge after dilation share an id (treated as one zone);
                    # crosswalks on different sides of an intersection stay separate.
                    n_labels, labels = cv2.connectedComponents(cached_dilated)
                    # Light per-component noise filter: drop specks < 200 px but
                    # keep all meaningful zebra-crossing slices.
                    min_comp = 200
                    kept = np.zeros(n_labels, dtype=np.int32)
                    next_id = 1
                    for lab in range(1, n_labels):
                        if int((labels == lab).sum()) >= min_comp:
                            kept[lab] = next_id
                            next_id += 1
                    cached_labels = kept[labels]
                    cached_area = area
                    cached_n_components = next_id - 1
                if cache_enabled:
                    mask_cache.append(cached_mask)
                    cur_mask_idx = len(mask_cache) - 1
            else:
                # yolo_seg backend:source 自行管理膨脹 + components
                labels_arr = crosswalk_source.get_labels(frame, idx)
                cached_labels = labels_arr
                cached_area = int((labels_arr > 0).sum())
                cached_n_components = int(labels_arr.max())
                # 由 labels 還原一張 binary mask;注意這個 mask 已是 post-dilate,
                # 下游若要再 dilate 需傳 dilate_px=0。
                cached_mask = (labels_arr > 0).astype(np.uint8) * 255
                cached_dilated = cached_mask
                if cache_enabled:
                    mask_cache.append(cached_mask)
                    cur_mask_idx = len(mask_cache) - 1
            last_seg_frame = idx

        # YOLO detection + ByteTrack (persistent IDs across stride calls)
        yolo_results = yolo.track(
            frame,
            persist=True,
            conf=conf,
            imgsz=imgsz,
            classes=list({PERSON_ID, *VEHICLE_IDS}),
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        ped_hits_by_comp: dict[int, list[int]] = {}  # comp_id -> list of tids
        veh_hits_by_comp: dict[int, list[tuple[int, float, float]]] = {}  # comp_id -> [(tid, d, s)]
        frame_dets: list[dict] = []

        if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            cls_arr = yolo_results.boxes.cls.cpu().numpy().astype(int)
            confs = yolo_results.boxes.conf.cpu().numpy()
            ids = yolo_results.boxes.id.cpu().numpy().astype(int)

            # Flag riders: a "person" bbox whose center/area is mostly inside a
            # motorcycle/bicycle bbox is the rider, not a pedestrian. Using
            # containment (not IoU) because rider bbox is usually taller than
            # the motorcycle bbox below it.
            TWO_WHEELER_CLASSES = {1, 3}  # bicycle, motorcycle
            rider_flags = [False] * len(boxes)
            for i, (bb_i, c_i) in enumerate(zip(boxes, cls_arr)):
                if c_i != PERSON_ID:
                    continue
                for j, (bb_j, c_j) in enumerate(zip(boxes, cls_arr)):
                    if i == j or c_j not in TWO_WHEELER_CLASSES:
                        continue
                    if is_rider(bb_i, bb_j,
                                contain_thresh=rider_contain_thresh,
                                foot_margin_px=rider_foot_margin_px):
                        rider_flags[i] = True
                        break

            for det_i, (bb, c, cf, tid) in enumerate(zip(boxes, cls_arr, confs, ids)):
                fx, fy = foot_point(bb)
                fx_c = max(0, min(W - 1, fx))
                fy_c = max(0, min(H - 1, fy))
                cx = 0.5 * (bb[0] + bb[2])
                cy = 0.5 * (bb[1] + bb[3])

                hit_ped = False
                hit_veh = False
                rider_flag = rider_flags[det_i] if det_i < len(rider_flags) else False

                if c == PERSON_ID:
                    # Enforce higher confidence for persons to suppress
                    # low-conf motorcycle-as-person misclassifications.
                    if float(cf) < person_conf:
                        prev_centers[int(tid)] = (float(cx), float(cy))
                        if cache_enabled:
                            frame_dets.append({
                                "cls": int(c), "tid": int(tid),
                                "xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                "hit_ped": False, "hit_veh": False,
                                "is_ego": False, "is_rider": False,
                                "low_conf": True,
                            })
                        continue
                    if rider_flag:
                        # Rider: not a pedestrian crossing; skip ped rule.
                        pass
                    elif cached_area > 0 and cached_labels is not None:
                        comp = _bbox_component(bb, cached_labels, strip_frac=0.15)
                        if comp > 0:
                            ped_hits_by_comp.setdefault(comp, []).append(int(tid))
                            hit_ped = True
                elif c in VEHICLE_IDS:
                    is_ego = exclude_ego and is_ego_vehicle(
                        bb, W, H,
                        bottom_px=ego_bottom_px,
                        min_width_frac=ego_min_width_frac,
                        min_height_frac=ego_min_height_frac,
                    )
                    if not is_ego and cached_area > 0 and cached_labels is not None:
                        comp = _bbox_component(bb, cached_labels, strip_frac=0.20)
                        if comp > 0:
                            # bbox bottom strip overlaps a crosswalk component —
                            # robust to the vehicle occluding its own crosswalk
                            prev = prev_centers.get(int(tid))
                            speed = 0.0 if prev is None else float(
                                np.hypot(cx - prev[0], cy - prev[1])
                            )
                            veh_hits_by_comp.setdefault(comp, []).append(
                                (int(tid), 0.0, speed))

                prev_centers[int(tid)] = (float(cx), float(cy))
                if cache_enabled:
                    frame_dets.append({
                        "cls": int(c),
                        "tid": int(tid),
                        "xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                        "hit_ped": hit_ped,
                        "hit_veh": False,  # filled in after rule eval below
                        "is_ego": bool(
                            exclude_ego and c in VEHICLE_IDS and is_ego_vehicle(
                                bb, W, H,
                                bottom_px=ego_bottom_px,
                                min_width_frac=ego_min_width_frac,
                                min_height_frac=ego_min_height_frac,
                            )
                        ),
                        "is_rider": bool(rider_flag),
                        "low_conf": False,
                    })

        # Rule: same-component requirement. A ped on crosswalk component X and a
        # moving vehicle on component Y only trigger if X == Y. This filters the
        # "different crosswalk at the same intersection" false alarm.
        shared_comps = set(ped_hits_by_comp.keys()) & set(veh_hits_by_comp.keys())
        triggered = False
        fired_ped_ids: set[int] = set()
        fired_veh_ids: set[int] = set()
        fired_comps: set[int] = set()
        min_d_all = 1e9
        max_s_all = 0.0
        for comp in shared_comps:
            moving = [(tid, d, s) for (tid, d, s) in veh_hits_by_comp[comp] if s >= moving_px]
            if not moving:
                continue
            triggered = True
            fired_comps.add(comp)
            fired_ped_ids.update(ped_hits_by_comp[comp])
            fired_veh_ids.update(tid for (tid, _, _) in moving)
            min_d_all = min(min_d_all, min(d for (_, d, _) in moving))
            max_s_all = max(max_s_all, max(s for (_, _, s) in moving))

        if triggered:
            frame_hits.append(
                FrameHit(
                    frame_idx=idx,
                    t_sec=idx / fps,
                    ped_track_ids=sorted(fired_ped_ids),
                    veh_track_ids=sorted(fired_veh_ids),
                    min_vehicle_distance_px=min_d_all,
                    vehicle_speed_px=max_s_all,
                )
            )
        if cache_enabled:
            # Backfill veh hit flag for detections whose tid fired the rule
            for d in frame_dets:
                if d["tid"] in fired_veh_ids:
                    d["hit_veh"] = True
                # Only surface ped HIT for peds that actually fired the rule
                # (reject peds on components that had no vehicle partner)
                if d["hit_ped"] and d["tid"] not in fired_ped_ids:
                    d["hit_ped"] = False
            frame_records.append(FrameRecord(
                idx=idx,
                mask_idx=cur_mask_idx if cached_area > 0 else -1,
                detections=frame_dets,
            ))

        # Preview
        if writer is not None and idx / fps <= preview_max_seconds:
            vis = frame.copy()
            # mask overlay (green)
            if cached_mask is not None:
                overlay = vis.copy()
                overlay[cached_mask > 0] = (0, 255, 0)
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
                # dilated edge (yellow)
                edge = cv2.Canny(cached_dilated, 50, 150)
                vis[edge > 0] = (0, 255, 255)
            # boxes
            if yolo_results.boxes is not None and yolo_results.boxes.id is not None:
                for bb, c, tid in zip(boxes, cls_arr, ids):
                    x1, y1, x2, y2 = [int(v) for v in bb]
                    is_ped = c == PERSON_ID
                    color = (0, 200, 255) if is_ped else (255, 100, 0)
                    if is_ped and int(tid) in ped_hits:
                        color = (0, 0, 255)
                    if (not is_ped) and any(int(tid) == t for (t, _, _) in moving_vehicles):
                        color = (0, 0, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{int(c)}#{int(tid)}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            tag = "HIT" if triggered else "."
            cv2.putText(vis, f"f={idx} t={idx/fps:.2f}s {tag}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)

        idx += 1
        if idx % 30 == 0:
            elapsed = time.monotonic() - t_start
            rate = idx / max(elapsed, 1e-6)
            eta = (max_frame - idx) / max(rate, 1e-6)
            log(f"[run] {idx}/{max_frame} ({100*idx/max_frame:5.1f}%) "
                f"{rate:.1f} fps, eta {eta:.0f}s, hits={len(frame_hits)}")
            if progress_cb is not None:
                progress_cb("analyzing", idx, max_frame, len(frame_hits))

    cap.release()
    if writer is not None:
        writer.release()

    # Aggregate consecutive frame hits
    events: list[Event] = []
    gap_frames = merge_gap_sec * fps
    cur: Event | None = None
    ped_set: set[int] = set()
    veh_set: set[int] = set()
    for h in frame_hits:
        if cur is None or (h.frame_idx - cur.end_frame) > gap_frames:
            if cur is not None:
                cur.ped_track_ids = sorted(ped_set)
                cur.veh_track_ids = sorted(veh_set)
                events.append(cur)
            cur = Event(
                start_frame=h.frame_idx,
                end_frame=h.frame_idx,
                start_sec=h.t_sec,
                end_sec=h.t_sec,
                min_distance_px=h.min_vehicle_distance_px,
                peak_speed_px=h.vehicle_speed_px,
            )
            ped_set = set(h.ped_track_ids)
            veh_set = set(h.veh_track_ids)
        else:
            cur.end_frame = h.frame_idx
            cur.end_sec = h.t_sec
            cur.min_distance_px = min(cur.min_distance_px, h.min_vehicle_distance_px)
            cur.peak_speed_px = max(cur.peak_speed_px, h.vehicle_speed_px)
            ped_set.update(h.ped_track_ids)
            veh_set.update(h.veh_track_ids)
    if cur is not None:
        cur.ped_track_ids = sorted(ped_set)
        cur.veh_track_ids = sorted(veh_set)
        events.append(cur)

    # Filter tiny events
    events = [
        e for e in events
        if (e.end_frame - e.start_frame + 1) >= min_event_frames
    ]

    report = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": max_frame,
        "duration_sec": max_frame / fps,
        "frame_hits": len(frame_hits),
        "events": [asdict(e) for e in events],
        "params": {
            "crosswalk_backend": crosswalk_backend,
            "mask_model": model_name,
            "yolo_weights": yolo_weights,
            "yolo_seg_weights": yolo_seg_weights,
            "yolo_seg_classes": yolo_seg_classes,
            "yolo_seg_conf": yolo_seg_conf,
            "yolo_seg_imgsz": yolo_seg_imgsz,
            "yolo_seg_min_component_px": yolo_seg_min_component_px,
            "stride": stride,
            "mask_every": mask_every,
            "dilate_px": dilate_px,
            "prox_px": prox_px,
            "moving_px": moving_px,
            "conf": conf,
            "imgsz": imgsz,
            "merge_gap_sec": merge_gap_sec,
            "min_event_frames": min_event_frames,
            "exclude_ego": exclude_ego,
            "ego_bottom_px": ego_bottom_px,
            "ego_min_width_frac": ego_min_width_frac,
            "ego_min_height_frac": ego_min_height_frac,
            "max_seconds": max_seconds,
            "min_mask_area_frac": min_mask_area_frac,
            "rider_contain_thresh": rider_contain_thresh,
            "rider_foot_margin_px": rider_foot_margin_px,
            "person_conf": person_conf,
            "mask_imgsz": mask_imgsz,
        },
    }
    if output_json is not None:
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"[done] report: {output_json}")
    log(f"[done] {len(frame_hits)} frame hits → {len(events)} events")
    for i, e in enumerate(events):
        log(f"  #{i+1}: frames {e.start_frame}-{e.end_frame}  "
            f"t=[{e.start_sec:.2f}s–{e.end_sec:.2f}s]  "
            f"min_dist={e.min_distance_px:.0f}px  peak_speed={e.peak_speed_px:.1f}px/f  "
            f"peds={e.ped_track_ids}  vehs={e.veh_track_ids}")
    if preview_video is not None:
        log(f"[done] preview: {preview_video}")
    if progress_cb is not None:
        progress_cb("analyzing", max_frame, max_frame, len(frame_hits))

    # ------- Second pass: write fully-annotated video -------
    if cache_enabled and annotated_out is not None:
        # yolo_seg 儲存在 mask_cache 裡的已經是 post-dilate,不可再 dilate。
        annot_dilate_px = 0 if crosswalk_backend == "yolo_seg" else dilate_px
        _write_annotated(
            video_path, annotated_out, frame_records, mask_cache,
            events, fps, W, H, max_frame, annot_dilate_px,
            progress_cb=progress_cb, cancel_event=cancel_event,
        )
        log(f"[done] annotated: {annotated_out}")

    if progress_cb is not None:
        progress_cb("done", max_frame, max_frame, len(frame_hits))

    return report


def _write_annotated(
    video_path: Path,
    out_path: Path,
    frame_records: list[FrameRecord],
    mask_cache: list[np.ndarray],
    events: list[Event],
    fps: float,
    W: int,
    H: int,
    max_frame: int,
    dilate_px: int,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Re-read source video and produce an annotated mp4 combining masks + boxes + events."""
    log(f"[annot] second pass → {out_path}")
    by_idx: dict[int, FrameRecord] = {r.idx: r for r in frame_records}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {out_path}")

    def active_event(fi: int) -> tuple[int, Event] | None:
        for i, e in enumerate(events):
            if e.start_frame <= fi <= e.end_frame:
                return i + 1, e
        return None

    def next_event(fi: int) -> tuple[int, Event] | None:
        for i, e in enumerate(events):
            if e.start_frame > fi:
                return i + 1, e
        return None

    last_rec: FrameRecord | None = None
    written = 0
    for idx in range(max_frame):
        if cancel_event is not None and cancel_event.is_set():
            cap.release()
            writer.release()
            raise Cancelled()
        ok, frame = cap.read()
        if not ok:
            break
        rec = by_idx.get(idx, last_rec)
        if idx in by_idx:
            last_rec = by_idx[idx]

        vis = frame.copy()

        # Mask overlay — each connected component (separate crosswalk) gets its
        # own color so the "different zebras at the same junction" scenario
        # is visually obvious.
        if rec is not None and rec.mask_idx >= 0:
            mask = mask_cache[rec.mask_idx]
            dilated = dilate_mask(mask, dilate_px) if dilate_px > 0 else mask
            n_labels, labels = cv2.connectedComponents(dilated)
            palette = [
                (0, 255, 0),    # green
                (0, 200, 255),  # cyan-ish (BGR) — actually amber-orange
                (255, 0, 255),  # magenta
                (0, 255, 255),  # yellow
                (255, 128, 0),  # azure
                (128, 255, 0),  # lime
            ]
            overlay = vis.copy()
            for lab in range(1, n_labels):
                color = palette[(lab - 1) % len(palette)]
                m = (labels == lab)
                overlay[m] = color
            vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            # Per-component dilated edge in the same palette
            for lab in range(1, n_labels):
                color = palette[(lab - 1) % len(palette)]
                edge = cv2.Canny(((labels == lab).astype(np.uint8)) * 255, 50, 150)
                vis[edge > 0] = color

        # Boxes
        if rec is not None:
            for d in rec.detections:
                if d.get("low_conf"):
                    continue
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                cls = d["cls"]
                tid = d["tid"]
                is_ped = cls == PERSON_ID
                if d.get("is_ego"):
                    color = (120, 120, 120)
                    label_prefix = "EGO"
                elif d.get("is_rider"):
                    color = (0, 200, 0)  # green — rider, not a ped
                    label_prefix = "RIDER"
                elif d["hit_ped"] or d["hit_veh"]:
                    color = (0, 0, 255)  # red — suspicious
                    label_prefix = "HIT"
                else:
                    color = (200, 180, 90) if is_ped else (180, 120, 60)
                    label_prefix = "ped" if is_ped else "veh"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label_prefix} {tid}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Event banner + timeline
        vis = _draw_event_banner(vis, idx, fps, active_event(idx), next_event(idx),
                                 events, max_frame)
        writer.write(vis)
        written += 1
        if written % 600 == 0:
            log(f"[annot] {written}/{max_frame} ({100*written/max_frame:.1f}%)")
        if progress_cb is not None and written % 120 == 0:
            progress_cb("annotating", written, max_frame, 0)

    cap.release()
    writer.release()
    log(f"[annot] wrote {written} frames")


def _draw_event_banner(
    img: np.ndarray,
    frame_idx: int,
    fps: float,
    active: tuple[int, Event] | None,
    nxt: tuple[int, Event] | None,
    events: list[Event],
    total: int,
) -> np.ndarray:
    H, W = img.shape[:2]
    out = img
    t_cur = frame_idx / fps

    # Bottom timeline strip
    strip_h = 34
    sy = H - strip_h
    ov = out.copy()
    cv2.rectangle(ov, (0, sy), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
    bar_y = sy + 20
    x0, x1 = 10, W - 10

    def x_of(fi: int) -> int:
        if total <= 1:
            return x0
        return int(x0 + (x1 - x0) * fi / (total - 1))

    cv2.line(out, (x0, bar_y + 3), (x1, bar_y + 3), (110, 110, 110), 1, cv2.LINE_AA)
    for e in events:
        a = x_of(e.start_frame)
        b = x_of(e.end_frame)
        if b - a < 2:
            b = a + 2
        cv2.rectangle(out, (a, bar_y), (b, bar_y + 6), (40, 40, 200), -1)
    px = x_of(frame_idx)
    cv2.line(out, (px, bar_y - 5), (px, bar_y + 11), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, _fmt_time(t_cur), (x0, sy + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    endlbl = _fmt_time(total / fps)
    (tw, _), _ = cv2.getTextSize(endlbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(out, endlbl, (x1 - tw, sy + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
    mid = f"{len(events)} events"
    (tw, _), _ = cv2.getTextSize(mid, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(out, mid, ((W - tw) // 2, sy + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    # Top banner
    if active is not None:
        ei, e = active
        pulse = 0.55 + 0.45 * abs(math.sin(frame_idx * 0.25))
        color = (int(40 * pulse), int(40 * pulse), int(255 * pulse))
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (W, 56), color, -1)
        out = cv2.addWeighted(ov, 0.8, out, 0.2, 0)
        cv2.putText(out, f"EVENT #{ei}  YIELDING SUSPECTED",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        sub = (f"t = {e.start_sec:.2f}s - {e.end_sec:.2f}s   "
               f"({e.end_sec - e.start_sec:.2f}s)   "
               f"peds={len(e.ped_track_ids)}  vehs={len(e.veh_track_ids)}")
        cv2.putText(out, sub, (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        # progress
        dur = max(1, e.end_frame - e.start_frame)
        pfrac = max(0.0, min(1.0, (frame_idx - e.start_frame) / dur))
        cv2.rectangle(out, (12, 52), (W - 12, 55), (80, 80, 80), -1)
        cv2.rectangle(out, (12, 52), (12 + int((W - 24) * pfrac), 55),
                      (255, 255, 255), -1)
        r = int(7 + 2 * math.sin(frame_idx * 0.25))
        cv2.circle(out, (W - 22, 22), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (W - 22, 22), r, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (W, 26), (30, 30, 30), -1)
        out = cv2.addWeighted(ov, 0.6, out, 0.4, 0)
        if nxt is not None:
            ni, ne = nxt
            gap = ne.start_sec - t_cur
            label = f"next EVENT #{ni} in {gap:5.1f}s  (t={ne.start_sec:.1f}s)"
        else:
            label = f"{len(events)} events total - no more ahead"
        cv2.putText(out, label, (12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "zero_shot_events.json")
    p.add_argument("--preview", type=Path, default=None,
                   help="If set, write an annotated preview video here.")
    p.add_argument("--preview-max-seconds", type=float, default=60.0)
    p.add_argument("--mask-model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic")
    p.add_argument("--yolo", default="yolo11n.pt",
                   help="YOLO weights path or model name (ultralytics will auto-download).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stride", type=int, default=1, help="Process every N-th frame.")
    p.add_argument("--mask-every", type=int, default=3,
                   help="Re-run Mask2Former every N processed frames. Static CCTV can use "
                        "a large value (60+); moving dashcam should stay low (1-5) so the "
                        "zebra-crossing mask tracks the camera's motion.")
    p.add_argument("--dilate-px", type=int, default=20)
    p.add_argument("--prox-px", type=int, default=0,
                   help="Vehicle is 'close' if foot-point is within this many px of the "
                        "dilated mask. Default 0 = must be ON the dilated mask (strict).")
    p.add_argument("--moving-px", type=float, default=2.0,
                   help="Per-frame centroid displacement (pixels) threshold to count as moving.")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--merge-gap-sec", type=float, default=0.6)
    p.add_argument("--min-event-frames", type=int, default=2)
    p.add_argument("--no-exclude-ego", action="store_true",
                   help="Disable dashcam ego-vehicle filtering (keep every YOLO vehicle).")
    p.add_argument("--ego-bottom-px", type=int, default=4)
    p.add_argument("--ego-min-width-frac", type=float, default=0.40)
    p.add_argument("--ego-min-height-frac", type=float, default=0.25)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Only process the first N seconds of the video.")
    p.add_argument("--min-mask-area-frac", type=float, default=0.005,
                   help="Ignore zebra masks smaller than this fraction of the frame (noise filter).")
    p.add_argument("--annotated-out", type=Path, default=None,
                   help="If set, write a full annotated mp4 with mask+boxes+event banners.")
    p.add_argument("--rider-contain-thresh", type=float, default=0.40,
                   help="Person is a RIDER if >= this fraction of the person bbox "
                        "(or of its lower half) overlaps a motorcycle/bicycle bbox.")
    p.add_argument("--rider-foot-margin-px", type=int, default=12,
                   help="A person is also treated as a RIDER if its foot-point falls "
                        "inside a motorcycle/bicycle bbox expanded by this many pixels.")
    p.add_argument("--person-conf", type=float, default=0.35,
                   help="Minimum confidence for class=person. Lower than this = dropped "
                        "(reduces motorcycle-misclassified-as-person noise).")
    p.add_argument("--mask-imgsz", type=int, default=0,
                   help="If > 0, resize frame so its short side is this many pixels "
                        "before Mask2Former inference (output mask is upscaled back). "
                        "Reduces GPU load / heat; 640 is a good value for 720p input.")
    p.add_argument("--crosswalk-backend", choices=["mask2former", "yolo_seg"],
                   default="mask2former",
                   help="Which crosswalk segmentation backend to use. "
                        "mask2former = GPU-heavy but accurate; yolo_seg = CPU-friendly "
                        "but depends on the quality of the supplied .pt weights.")
    p.add_argument("--yolo-seg-weights", type=str, default="",
                   help="Path to YOLO-seg .pt weights for crosswalk segmentation "
                        "(required when --crosswalk-backend=yolo_seg).")
    p.add_argument("--yolo-seg-classes", type=str, default="",
                   help="Comma-separated list of class names from the YOLO-seg model "
                        "to treat as crosswalk. Empty = accept all classes "
                        "(for single-class models).")
    p.add_argument("--yolo-seg-conf", type=float, default=0.25)
    p.add_argument("--yolo-seg-imgsz", type=int, default=640)
    p.add_argument("--yolo-seg-min-component-px", type=int, default=200,
                   help="膨脹 + connected-components 後,小於此像素數的 component 丟掉。"
                        "yolo_seg_baseline preset 設 80 抓遠處小斑馬線;Mask2Former 路徑固定 200。")
    args = p.parse_args()

    if args.crosswalk_backend == "yolo_seg" and args.preview is not None:
        # --preview 路徑裡有 bug 會炸(ped_hits / moving_vehicles 未定義;mask2former
        # 也一樣),短期跳過
        print("[warn] --preview 暫時在 yolo_seg 下忽略(已知 bug,日後修)", flush=True)
        args.preview = None

    yolo_seg_classes: list[str] | None = None
    if args.yolo_seg_classes.strip():
        yolo_seg_classes = [s.strip() for s in args.yolo_seg_classes.split(",") if s.strip()]

    # 讓 zebraguard package 能被 import(當 repo 未安裝時)
    import sys as _sys
    _src = Path(__file__).resolve().parent.parent / "src"
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))

    run(
        args.video,
        model_name=args.mask_model,
        yolo_weights=args.yolo,
        device_str=args.device,
        stride=args.stride,
        mask_every=args.mask_every,
        dilate_px=args.dilate_px,
        prox_px=args.prox_px,
        moving_px=args.moving_px,
        conf=args.conf,
        imgsz=args.imgsz,
        merge_gap_sec=args.merge_gap_sec,
        min_event_frames=args.min_event_frames,
        output_json=args.out,
        preview_video=args.preview,
        preview_max_seconds=args.preview_max_seconds,
        exclude_ego=not args.no_exclude_ego,
        ego_bottom_px=args.ego_bottom_px,
        ego_min_width_frac=args.ego_min_width_frac,
        ego_min_height_frac=args.ego_min_height_frac,
        max_seconds=args.max_seconds,
        min_mask_area_frac=args.min_mask_area_frac,
        annotated_out=args.annotated_out,
        rider_contain_thresh=args.rider_contain_thresh,
        rider_foot_margin_px=args.rider_foot_margin_px,
        person_conf=args.person_conf,
        mask_imgsz=args.mask_imgsz,
        crosswalk_backend=args.crosswalk_backend,
        yolo_seg_weights=args.yolo_seg_weights,
        yolo_seg_classes=yolo_seg_classes,
        yolo_seg_conf=args.yolo_seg_conf,
        yolo_seg_imgsz=args.yolo_seg_imgsz,
        yolo_seg_min_component_px=args.yolo_seg_min_component_px,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
