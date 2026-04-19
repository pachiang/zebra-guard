#!/usr/bin/env bash
# v5_baseline: the known-good config used to produce zs_taiwan10_v5.json
# on 2026-04-19 for taiwan_yield_720p.mp4 (first 600s). 31 events / 322 hits.
# All long-flag values below mirror the params block saved inside the JSON output,
# so any future run that logs these flags can be diffed against this file.

set -euo pipefail

.venv/Scripts/python.exe scripts/zero_shot_detect.py \
  --video "${1:-taiwan_yield_720p.mp4}" \
  --out "${2:-zs_out.json}" \
  --annotated-out "${3:-zs_out_annotated.mp4}" \
  --stride 3 \
  --mask-every 5 \
  --dilate-px 20 \
  --prox-px 0 \
  --moving-px 2.0 \
  --conf 0.20 \
  --imgsz 640 \
  --merge-gap-sec 0.6 \
  --min-event-frames 2 \
  --ego-bottom-px 4 \
  --ego-min-width-frac 0.40 \
  --ego-min-height-frac 0.25 \
  --max-seconds 600 \
  --min-mask-area-frac 0.005 \
  --rider-contain-thresh 0.40 \
  --rider-foot-margin-px 12 \
  --person-conf 0.35
