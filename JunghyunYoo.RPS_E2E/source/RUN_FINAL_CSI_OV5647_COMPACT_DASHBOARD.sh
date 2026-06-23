#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

python -u pi_rps_window_dashboard_compact480.py \
  --model ../models/FINAL_best_strong_aug_ptq_full_int8_io.tflite \
  --camera 0 \
  --camera-backend rpicam \
  --capture-width 640 \
  --capture-height 480 \
  --capture-fps 58 \
  --interpreter-backend tensorflow \
  --num-threads 4 \
  --display-width 800 \
  --display-height 480 \
  --fullscreen \
  --threshold 0.65 \
  --stable-frames 8 \
  --mirror
