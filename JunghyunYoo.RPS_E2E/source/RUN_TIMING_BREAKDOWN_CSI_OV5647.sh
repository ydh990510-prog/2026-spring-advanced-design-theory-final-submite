#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -u pi_timing_breakdown_csi_rpicam_tf4.py \
  --model ../models/FINAL_best_strong_aug_ptq_full_int8_io.tflite \
  --camera 0 \
  --camera-backend rpicam \
  --capture-width 640 \
  --capture-height 480 \
  --capture-fps 58 \
  --runs 500 \
  --warmup 50 \
  --dashboard-draw \
  --json-out timing_breakdown_csi_ov5647_dashboard_draw.json
